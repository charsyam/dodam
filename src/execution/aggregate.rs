use std::collections::HashSet;
use std::collections::hash_map::Entry;
use std::sync::{Arc, mpsc};
use std::time::Instant;

use arrow::array::{
    Array, ArrayRef, BooleanArray, Date32Array, Date64Array, Decimal128Array, DictionaryArray,
    Float64Array, Int32Array, Int64Array, StringArray, TimestampMillisecondArray, UInt64Array,
};
use arrow::compute::kernels::aggregate::{max, max_string, min, min_string, sum};
use arrow::datatypes::{DataType, Field, Int32Type, Schema, TimeUnit};
use arrow::record_batch::RecordBatch;
use arrow_row::{OwnedRow, RowConverter, SortField};

use crate::error::{DodamError, Result};
use crate::execution::logical::{
    AggregateExpr, AggregateMetrics, AggregateResult, AggregateValue, ComparisonOp, Expr,
    GroupAggregateResult, GroupValue, LiteralValue,
};
use crate::execution::metrics::SendableBatchStream;
use crate::execution::physical::column_index;
use crate::hash::FastHashMap as AggregateHashMap;
use crate::vector::dictionary_i32_string_values;
use crate::vector::{BatchView, DictionaryStringValues, I32VectorView, I64VectorView};

const SMALL_GROUP_LINEAR_LIMIT: usize = 8;
const TWO_UTF8_SMALL_GROUP_LIMIT: usize = 8;
const DENSE_I32_GROUP_INDEX_MAX_SLOTS: usize = 65_536;
const DENSE_U32_GROUP_INDEX_MAX_SLOTS: usize = 65_536;

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

fn dense_i32_group_index_max_slots() -> usize {
    std::env::var("DODAM_DENSE_I32_GROUP_INDEX_MAX_SLOTS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DENSE_I32_GROUP_INDEX_MAX_SLOTS)
}

fn dense_u32_group_index_max_slots() -> usize {
    std::env::var("DODAM_DENSE_U32_GROUP_INDEX_MAX_SLOTS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DENSE_U32_GROUP_INDEX_MAX_SLOTS)
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

pub(crate) struct CoalesceKeyCountSumCollector {
    plan: CoalesceKeyCountSumPlan,
    bound_plan: Option<BoundCoalesceKeyCountSumPlan>,
    groups: Option<CoalesceKeyCountSumGroups>,
    metrics: AggregateMetrics,
    bind_nanos: u64,
    reader_nanos: u64,
    update_nanos: u64,
}

impl CoalesceKeyCountSumCollector {
    pub(crate) fn new(group_keys: &[GroupKeyExpr], aggregates: &[AggregateExpr]) -> Option<Self> {
        Some(Self {
            plan: CoalesceKeyCountSumPlan::new(group_keys, aggregates)?,
            bound_plan: None,
            groups: None,
            metrics: AggregateMetrics::default(),
            bind_nanos: 0,
            reader_nanos: 0,
            update_nanos: 0,
        })
    }

    pub(crate) fn consume_batch(&mut self, batch: &RecordBatch) -> Result<()> {
        if batch.num_rows() == 0 {
            return Ok(());
        }
        self.metrics.batches += 1;
        self.metrics.rows += batch.num_rows();
        let bound = match &self.bound_plan {
            Some(bound) => bound,
            None => {
                let started = Instant::now();
                self.bound_plan = Some(self.plan.bind(batch)?);
                self.bind_nanos = self.bind_nanos.saturating_add(elapsed_nanos_u64(started));
                self.bound_plan
                    .as_ref()
                    .expect("bound coalesce aggregate plan")
            }
        };
        let first = batch.column(bound.first);
        let second = bound.second.map(|index| batch.column(index));
        let coalesce = batch.column(bound.coalesce);
        let sum = batch.column(bound.sum);
        let started = Instant::now();
        let Some(reader) = CoalesceKeyCountSumReader::new(first, second, coalesce, sum) else {
            return Err(DodamError::UnsupportedSql(
                "group expression view aggregate requires integer/date leading keys, coalesce(utf8,literal), and integer sum inputs"
                    .to_string(),
            ));
        };
        self.reader_nanos = self.reader_nanos.saturating_add(elapsed_nanos_u64(started));
        let groups = self.groups.get_or_insert_with(|| {
            CoalesceKeyCountSumGroups::new(
                self.plan.leading_keys.len(),
                reader.dense_leading_range(batch.num_rows()),
            )
        });
        let started = Instant::now();
        reader.update_groups(groups, batch.num_rows(), self.plan.fallback.as_str());
        self.update_nanos = self.update_nanos.saturating_add(elapsed_nanos_u64(started));
        Ok(())
    }

    pub(crate) fn merge_partials(
        partials: Vec<Self>,
        fragments: usize,
        aggregates: &[AggregateExpr],
    ) -> Result<AggregateMetrics> {
        Self::merge_partials_with_order(partials, fragments, aggregates, true)
    }

    pub(crate) fn merge_partials_with_order(
        partials: Vec<Self>,
        fragments: usize,
        aggregates: &[AggregateExpr],
        ordered: bool,
    ) -> Result<AggregateMetrics> {
        let merge_started = Instant::now();
        let mut merged_groups: Option<CoalesceKeyCountSumGroups> = None;
        let mut metrics = AggregateMetrics {
            fragments,
            ..AggregateMetrics::default()
        };
        let mut update_nanos = 0_u64;
        for partial in partials {
            metrics.batches = metrics.batches.saturating_add(partial.metrics.batches);
            metrics.rows = metrics.rows.saturating_add(partial.metrics.rows);
            update_nanos = update_nanos.saturating_add(partial.update_nanos);
            let Some(groups) = partial.groups else {
                continue;
            };
            match &mut merged_groups {
                Some(merged) => merged.merge_from(groups)?,
                None => merged_groups = Some(groups),
            }
        }
        metrics.aggregate_nanos = update_nanos;
        metrics.aggregate_merge_nanos = elapsed_nanos(merge_started);
        metrics.groups = merged_groups
            .map(|groups| groups.finish_with_order(aggregates, ordered))
            .unwrap_or_default();
        Ok(metrics)
    }
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
    let profile = coalesce_key_count_sum_profile_enabled();
    let total_started = profile.then(Instant::now);
    let mut bind_nanos = 0_u64;
    let mut reader_nanos = 0_u64;
    let mut update_nanos = 0_u64;
    let mut finish_nanos = 0_u64;
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
                let started = Instant::now();
                bound_plan = Some(plan.bind(&batch)?);
                bind_nanos = bind_nanos.saturating_add(elapsed_nanos_u64(started));
                bound_plan.as_ref().expect("bound coalesce aggregate plan")
            }
        };
        let first = batch.column(bound.first);
        let second = bound.second.map(|index| batch.column(index));
        let coalesce = batch.column(bound.coalesce);
        let sum = batch.column(bound.sum);
        let started = Instant::now();
        let Some(reader) = CoalesceKeyCountSumReader::new(first, second, coalesce, sum) else {
            return Err(DodamError::UnsupportedSql(
                "group expression view aggregate requires integer/date leading keys, coalesce(utf8,literal), and integer sum inputs"
                    .to_string(),
            ));
        };
        reader_nanos = reader_nanos.saturating_add(elapsed_nanos_u64(started));
        let groups = groups.get_or_insert_with(|| {
            CoalesceKeyCountSumGroups::new(
                plan.leading_keys.len(),
                reader.dense_leading_range(batch.num_rows()),
            )
        });
        let started = Instant::now();
        reader.update_groups(groups, batch.num_rows(), plan.fallback.as_str());
        update_nanos = update_nanos.saturating_add(elapsed_nanos_u64(started));
    }
    let started = Instant::now();
    metrics.groups = groups
        .map(|groups| groups.finish(aggregates))
        .unwrap_or_default();
    finish_nanos = finish_nanos.saturating_add(elapsed_nanos_u64(started));
    if profile {
        eprintln!(
            "[dodam:coalesce-agg-profile] total={:.3}ms bind={:.3}ms reader={:.3}ms update={:.3}ms finish={:.3}ms batches={} rows={} groups={}",
            nanos_to_millis_f64(total_started.map(elapsed_nanos_u64).unwrap_or(0)),
            nanos_to_millis_f64(bind_nanos),
            nanos_to_millis_f64(reader_nanos),
            nanos_to_millis_f64(update_nanos),
            nanos_to_millis_f64(finish_nanos),
            metrics.batches,
            metrics.rows,
            metrics.groups.len(),
        );
    }
    Ok(metrics)
}

fn coalesce_key_count_sum_profile_enabled() -> bool {
    std::env::var("DODAM_COALESCE_AGG_PROFILE")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn elapsed_nanos_u64(started: Instant) -> u64 {
    started.elapsed().as_nanos().min(u64::MAX as u128) as u64
}

fn nanos_to_millis_f64(nanos: u64) -> f64 {
    nanos as f64 / 1_000_000.0
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
        coalesce: CoalesceUtf8Input<'a>,
        sum: Int64LikeArray<'a>,
    },
    TwoInt64 {
        first: &'a Int64Array,
        coalesce: CoalesceUtf8Input<'a>,
        sum: Int64LikeArray<'a>,
    },
    ThreeInt32Date {
        first: &'a Int32Array,
        second: &'a Date32Array,
        coalesce: CoalesceUtf8Input<'a>,
        sum: Int64LikeArray<'a>,
    },
    ThreeInt64Date {
        first: &'a Int64Array,
        second: &'a Date32Array,
        coalesce: CoalesceUtf8Input<'a>,
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
        let coalesce = CoalesceUtf8Input::new(coalesce)?;
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
                third: coalesce.key(row, fallback),
            },
            Self::TwoInt64 {
                first, coalesce, ..
            } => CoalesceKeyBorrowed {
                first: first.is_valid(row).then(|| first.value(row)),
                second: None,
                third: coalesce.key(row, fallback),
            },
            Self::ThreeInt32Date {
                first,
                second,
                coalesce,
                ..
            } => CoalesceKeyBorrowed {
                first: first.is_valid(row).then(|| i64::from(first.value(row))),
                second: second.is_valid(row).then(|| second.value(row)),
                third: coalesce.key(row, fallback),
            },
            Self::ThreeInt64Date {
                first,
                second,
                coalesce,
                ..
            } => CoalesceKeyBorrowed {
                first: first.is_valid(row).then(|| first.value(row)),
                second: second.is_valid(row).then(|| second.value(row)),
                third: coalesce.key(row, fallback),
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
            Self::TwoInt32 {
                first,
                coalesce,
                sum,
            } if first.null_count() == 0 => {
                if let Some(mut cache) = CoalesceStringIdCache::new(coalesce, groups, fallback) {
                    for row in 0..row_count {
                        let string_id = cache.string_id(coalesce, groups, row);
                        groups.update_two_non_null_string_id(
                            first.value(row).into(),
                            string_id,
                            sum.value(row),
                        );
                    }
                } else {
                    for row in 0..row_count {
                        groups.update_two_non_null(
                            i64::from(first.value(row)),
                            coalesce
                                .key(row, fallback)
                                .expect("coalesce key is non-null"),
                            sum.value(row),
                        );
                    }
                }
            }
            Self::TwoInt64 {
                first,
                coalesce,
                sum,
            } if first.null_count() == 0 => {
                if let Some(mut cache) = CoalesceStringIdCache::new(coalesce, groups, fallback) {
                    for row in 0..row_count {
                        let string_id = cache.string_id(coalesce, groups, row);
                        groups.update_two_non_null_string_id(
                            first.value(row),
                            string_id,
                            sum.value(row),
                        );
                    }
                } else {
                    for row in 0..row_count {
                        groups.update_two_non_null(
                            first.value(row),
                            coalesce
                                .key(row, fallback)
                                .expect("coalesce key is non-null"),
                            sum.value(row),
                        );
                    }
                }
            }
            Self::ThreeInt32Date {
                first,
                second,
                coalesce,
                sum,
            } if first.null_count() == 0 && second.null_count() == 0 => {
                if let Some(mut cache) = CoalesceStringIdCache::new(coalesce, groups, fallback) {
                    for row in 0..row_count {
                        let string_id = cache.string_id(coalesce, groups, row);
                        groups.update_three_non_null_string_id(
                            first.value(row).into(),
                            second.value(row),
                            string_id,
                            sum.value(row),
                        );
                    }
                } else {
                    for row in 0..row_count {
                        groups.update_three_non_null(
                            i64::from(first.value(row)),
                            second.value(row),
                            coalesce
                                .key(row, fallback)
                                .expect("coalesce key is non-null"),
                            sum.value(row),
                        );
                    }
                }
            }
            Self::ThreeInt64Date {
                first,
                second,
                coalesce,
                sum,
            } if first.null_count() == 0 && second.null_count() == 0 => {
                if let Some(mut cache) = CoalesceStringIdCache::new(coalesce, groups, fallback) {
                    for row in 0..row_count {
                        let string_id = cache.string_id(coalesce, groups, row);
                        groups.update_three_non_null_string_id(
                            first.value(row),
                            second.value(row),
                            string_id,
                            sum.value(row),
                        );
                    }
                } else {
                    for row in 0..row_count {
                        groups.update_three_non_null(
                            first.value(row),
                            second.value(row),
                            coalesce
                                .key(row, fallback)
                                .expect("coalesce key is non-null"),
                            sum.value(row),
                        );
                    }
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
            Self::TwoInt32 { first, .. } => {
                let (first_min, first_max) = int32_non_null_min_max_as_i64(first, row_count)?;
                (first_min, first_max, 0, 0)
            }
            Self::TwoInt64 { first, .. } => {
                let (first_min, first_max) = int64_non_null_min_max(first, row_count)?;
                (first_min, first_max, 0, 0)
            }
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

#[derive(Clone, Copy)]
enum CoalesceUtf8Input<'a> {
    Utf8(&'a StringArray),
    DictionaryI32 {
        keys: &'a DictionaryArray<Int32Type>,
        values: DictionaryStringValues<'a>,
    },
}

impl<'a> CoalesceUtf8Input<'a> {
    fn new(array: &'a ArrayRef) -> Option<Self> {
        if let Some(values) = array.as_any().downcast_ref::<StringArray>() {
            return Some(Self::Utf8(values));
        }
        if let DataType::Dictionary(key, value) = array.data_type()
            && matches!(&**key, DataType::Int32)
            && matches!(&**value, DataType::Utf8 | DataType::LargeUtf8)
        {
            let keys = array
                .as_any()
                .downcast_ref::<DictionaryArray<Int32Type>>()?;
            return Some(Self::DictionaryI32 {
                keys,
                values: dictionary_i32_string_values(keys)?,
            });
        }
        None
    }

    fn key(self, row: usize, fallback: &'a str) -> Option<&'a str> {
        match self {
            Self::Utf8(values) => coalesce_key(values, row, fallback),
            Self::DictionaryI32 { keys, values } => {
                if keys.is_null(row) {
                    return Some(fallback);
                }
                let Ok(id) = usize::try_from(keys.keys().value(row)) else {
                    return Some(fallback);
                };
                Some(
                    std::str::from_utf8(values.value_bytes(id))
                        .expect("Arrow dictionary Utf8 value should be valid UTF8"),
                )
            }
        }
    }

    fn dictionary_key(self, row: usize) -> Option<i32> {
        match self {
            Self::DictionaryI32 { keys, .. } if keys.is_valid(row) => Some(keys.keys().value(row)),
            Self::DictionaryI32 { .. } | Self::Utf8(_) => None,
        }
    }

    fn dictionary_value(self, id: usize) -> Option<&'a str> {
        match self {
            Self::DictionaryI32 { values, .. } => Some(
                std::str::from_utf8(values.value_bytes(id))
                    .expect("Arrow dictionary Utf8 value should be valid UTF8"),
            ),
            Self::Utf8(_) => None,
        }
    }

    fn utf8_value(self, row: usize) -> Option<&'a str> {
        match self {
            Self::Utf8(values) if values.is_valid(row) => Some(values.value(row)),
            Self::Utf8(_) | Self::DictionaryI32 { .. } => None,
        }
    }
}

enum CoalesceStringIdCache<'a> {
    Utf8 {
        fallback_id: u32,
        ids: AggregateHashMap<&'a str, u32>,
    },
    Dictionary {
        fallback_id: u32,
        ids: Vec<Option<u32>>,
    },
}

impl<'a> CoalesceStringIdCache<'a> {
    fn new(
        coalesce: &CoalesceUtf8Input<'a>,
        groups: &mut CoalesceKeyCountSumGroups,
        fallback: &str,
    ) -> Option<Self> {
        let fallback_id = groups.string_id(fallback);
        match coalesce {
            CoalesceUtf8Input::Utf8(_) => Some(Self::Utf8 {
                fallback_id,
                ids: AggregateHashMap::default(),
            }),
            CoalesceUtf8Input::DictionaryI32 { values, .. } => Some(Self::Dictionary {
                fallback_id,
                ids: vec![None; values.len()],
            }),
        }
    }

    fn string_id(
        &mut self,
        coalesce: &CoalesceUtf8Input<'a>,
        groups: &mut CoalesceKeyCountSumGroups,
        row: usize,
    ) -> u32 {
        match self {
            Self::Utf8 { fallback_id, ids } => {
                let Some(value) = coalesce.utf8_value(row) else {
                    return *fallback_id;
                };
                if let Some(string_id) = ids.get(value).copied() {
                    return string_id;
                }
                let string_id = groups.string_id(value);
                ids.insert(value, string_id);
                string_id
            }
            Self::Dictionary { fallback_id, ids } => {
                let Some(id) = coalesce.dictionary_key(row) else {
                    return *fallback_id;
                };
                let Ok(index) = usize::try_from(id) else {
                    return *fallback_id;
                };
                if let Some(string_id) = ids.get(index).and_then(|id| *id) {
                    return string_id;
                }
                let Some(value) = coalesce.dictionary_value(index) else {
                    return *fallback_id;
                };
                let string_id = groups.string_id(value);
                if let Some(slot) = ids.get_mut(index) {
                    *slot = Some(string_id);
                }
                string_id
            }
        }
    }
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

    fn string_id(&mut self, value: &str) -> u32 {
        if let Some(string_id) = self.third_string_ids.get(value).copied() {
            string_id
        } else {
            let string_id =
                u32::try_from(self.third_strings.len()).expect("too many string groups");
            self.third_string_ids.insert(value.to_string(), string_id);
            self.third_strings.push(value.to_string());
            string_id
        }
    }

    fn update(&mut self, key: CoalesceKeyBorrowed<'_>, sum: Option<i64>) {
        let string_id = match key.third {
            Some(value) => Some(self.string_id(value)),
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
                    self.groups.push(CoalesceKeyCountSumGroup::new(
                        key.first,
                        key.second,
                        Some(string_id),
                    ));
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
                        .push(CoalesceKeyCountSumGroup::new(key.first, key.second, None));
                    group_id
                }
            }
        };
        self.groups[group_id].update(sum);
    }

    fn update_three_non_null(&mut self, first: i64, second: i32, third: &str, sum: Option<i64>) {
        let string_id = self.string_id(third);
        self.update_three_non_null_string_id(first, second, string_id, sum);
    }

    fn update_three_non_null_string_id(
        &mut self,
        first: i64,
        second: i32,
        string_id: u32,
        sum: Option<i64>,
    ) {
        let third_index = self.index.third_groups_three_non_null(first, second);
        let group_id = if let Some(group_id) = third_index.non_null.get(string_id) {
            group_id
        } else {
            let group_id = self.groups.len();
            third_index.non_null.insert(string_id, group_id);
            self.groups.push(CoalesceKeyCountSumGroup::new(
                Some(first),
                Some(second),
                Some(string_id),
            ));
            group_id
        };
        self.groups[group_id].update(sum);
    }

    fn update_two_non_null(&mut self, first: i64, third: &str, sum: Option<i64>) {
        let string_id = self.string_id(third);
        self.update_two_non_null_string_id(first, string_id, sum);
    }

    fn update_two_non_null_string_id(&mut self, first: i64, string_id: u32, sum: Option<i64>) {
        let third_index = self.index.third_groups_two_non_null(first);
        let group_id = if let Some(group_id) = third_index.non_null.get(string_id) {
            group_id
        } else {
            let group_id = self.groups.len();
            third_index.non_null.insert(string_id, group_id);
            self.groups.push(CoalesceKeyCountSumGroup::new(
                Some(first),
                None,
                Some(string_id),
            ));
            group_id
        };
        self.groups[group_id].update(sum);
    }

    fn merge_from(&mut self, other: CoalesceKeyCountSumGroups) -> Result<()> {
        if self.key_len != other.key_len {
            return Err(DodamError::UnsupportedSql(
                "coalesce aggregate partial key shape mismatch".to_string(),
            ));
        }
        for group in other.groups {
            let third_string_id = match group.third_string_id {
                Some(id) => {
                    let value = other.third_strings.get(id as usize).ok_or_else(|| {
                        DodamError::UnsupportedSql(
                            "coalesce aggregate partial string id mismatch".to_string(),
                        )
                    })?;
                    Some(self.string_id(value))
                }
                None => None,
            };
            self.merge_group_counts(
                group.first,
                group.second,
                third_string_id,
                group.count,
                group.sum,
                group.sum_count,
            );
        }
        Ok(())
    }

    fn merge_group_counts(
        &mut self,
        first: Option<i64>,
        second: Option<i32>,
        third_string_id: Option<u32>,
        count: u64,
        sum: i64,
        sum_count: u64,
    ) {
        let third_index = self.index.third_groups(first, second);
        let group_id = match third_string_id {
            Some(string_id) => {
                if let Some(group_id) = third_index.non_null.get(string_id) {
                    group_id
                } else {
                    let group_id = self.groups.len();
                    third_index.non_null.insert(string_id, group_id);
                    self.groups.push(CoalesceKeyCountSumGroup::new(
                        first,
                        second,
                        Some(string_id),
                    ));
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
                        .push(CoalesceKeyCountSumGroup::new(first, second, None));
                    group_id
                }
            }
        };
        self.groups[group_id].merge_counts(count, sum, sum_count);
    }

    fn finish(self, aggregates: &[AggregateExpr]) -> Vec<GroupAggregateResult> {
        self.finish_with_order(aggregates, true)
    }

    fn finish_with_order(
        self,
        aggregates: &[AggregateExpr],
        ordered: bool,
    ) -> Vec<GroupAggregateResult> {
        let Self {
            key_len,
            index,
            third_string_ids,
            third_strings,
            groups,
        } = self;
        if ordered {
            match index {
                CoalesceLeadingIndex::DenseTwo {
                    range,
                    slots,
                    fallback,
                } if key_len == 2 && fallback.is_empty() => {
                    return finish_dense_two_ordered_groups(
                        range,
                        slots,
                        third_strings,
                        groups,
                        aggregates,
                    );
                }
                CoalesceLeadingIndex::DenseThree {
                    range,
                    slots,
                    fallback,
                } if key_len == 3 && fallback.is_empty() => {
                    return finish_dense_three_ordered_groups(
                        range,
                        slots,
                        third_strings,
                        groups,
                        aggregates,
                    );
                }
                index => {
                    let _ = index;
                }
            }
        } else {
            let _ = index;
        }
        let _ = third_string_ids;
        let _ = third_strings;
        let mut groups = groups
            .into_iter()
            .map(|group| group.finish(key_len, &third_strings, aggregates))
            .collect::<Vec<_>>();
        if ordered {
            groups.sort_by(|left, right| compare_group_keys(&left.keys, &right.keys));
        }
        groups
    }
}

fn finish_dense_two_ordered_groups(
    range: DenseLeadingRange,
    slots: Vec<CoalesceThirdGroups>,
    third_strings: Vec<String>,
    groups: Vec<CoalesceKeyCountSumGroup>,
    aggregates: &[AggregateExpr],
) -> Vec<GroupAggregateResult> {
    let mut groups = groups.into_iter().map(Some).collect::<Vec<_>>();
    let mut output = Vec::with_capacity(groups.len());
    for first_offset in 0..range.first_len {
        let third_groups = &slots[first_offset];
        if let Some(group_id) = third_groups.null_group
            && let Some(group) = groups[group_id].take()
        {
            output.push(group.finish(2, &third_strings, aggregates));
        }
        let mut non_null = third_groups.non_null.iter().collect::<Vec<_>>();
        non_null.sort_by(|(left, _), (right, _)| {
            third_strings[*left as usize].cmp(&third_strings[*right as usize])
        });
        for (_, group_id) in non_null {
            if let Some(group) = groups[group_id].take() {
                output.push(group.finish(2, &third_strings, aggregates));
            }
        }
    }
    output
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
                output.push(group.finish(3, &third_strings, aggregates));
            }
            let mut non_null = third_groups.non_null.iter().collect::<Vec<_>>();
            non_null.sort_by(|(left, _), (right, _)| {
                third_strings[*left as usize].cmp(&third_strings[*right as usize])
            });
            for (_, group_id) in non_null {
                if let Some(group) = groups[group_id].take() {
                    output.push(group.finish(3, &third_strings, aggregates));
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
    DenseTwo {
        range: DenseLeadingRange,
        slots: Vec<CoalesceThirdGroups>,
        fallback: AggregateHashMap<Option<i64>, CoalesceSecondGroups>,
    },
    DenseThree {
        range: DenseLeadingRange,
        slots: Vec<CoalesceThirdGroups>,
        fallback: AggregateHashMap<Option<i64>, CoalesceSecondGroups>,
    },
}

impl CoalesceLeadingIndex {
    fn new(leading_key_count: usize, dense_range: Option<DenseLeadingRange>) -> Self {
        if leading_key_count == 1
            && let Some(range) = dense_range
        {
            return Self::DenseTwo {
                range,
                slots: (0..range.first_len)
                    .map(|_| CoalesceThirdGroups::default())
                    .collect(),
                fallback: AggregateHashMap::default(),
            };
        }
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
            Self::DenseTwo {
                range,
                slots,
                fallback,
            } => {
                if second.is_none()
                    && let Some(first) = first
                    && let Some(slot) = dense_first_slot(*range, first)
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

    fn third_groups_two_non_null(&mut self, first: i64) -> &mut CoalesceThirdGroups {
        match self {
            Self::Hash(index) => index
                .entry(Some(first))
                .or_default()
                .index
                .entry(None)
                .or_default(),
            Self::DenseTwo {
                range,
                slots,
                fallback,
            } => {
                if let Some(slot) = dense_first_slot(*range, first) {
                    return &mut slots[slot];
                }
                fallback
                    .entry(Some(first))
                    .or_default()
                    .index
                    .entry(None)
                    .or_default()
            }
            Self::DenseThree { fallback, .. } => fallback
                .entry(Some(first))
                .or_default()
                .index
                .entry(None)
                .or_default(),
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
            Self::DenseTwo { fallback, .. } => fallback
                .entry(Some(first))
                .or_default()
                .index
                .entry(Some(second))
                .or_default(),
        }
    }
}

fn dense_first_slot(range: DenseLeadingRange, first: i64) -> Option<usize> {
    let first_offset = usize::try_from(first.checked_sub(range.first_min)?).ok()?;
    (first_offset < range.first_len).then_some(first_offset)
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

#[derive(Default)]
struct CoalesceSecondGroups {
    index: AggregateHashMap<Option<i32>, CoalesceThirdGroups>,
}

#[derive(Default)]
struct CoalesceThirdGroups {
    non_null: DenseU32GroupIndex,
    null_group: Option<usize>,
}

enum DenseU32GroupIndex {
    Small(Vec<(u32, usize)>),
    Dense(Vec<Option<usize>>),
    Hash(AggregateHashMap<u32, usize>),
}

impl Default for DenseU32GroupIndex {
    fn default() -> Self {
        Self::Small(Vec::new())
    }
}

impl DenseU32GroupIndex {
    fn get(&self, key: u32) -> Option<usize> {
        match self {
            Self::Small(groups) => groups
                .iter()
                .find_map(|(candidate, group_id)| (*candidate == key).then_some(*group_id)),
            Self::Dense(slots) => slots.get(key as usize).copied().flatten(),
            Self::Hash(groups) => groups.get(&key).copied(),
        }
    }

    fn insert(&mut self, key: u32, group_id: usize) {
        match self {
            Self::Small(groups) if groups.len() < small_group_linear_limit() => {
                groups.push((key, group_id));
            }
            Self::Small(groups) => {
                if let Some(mut slots) = dense_u32_slots_for_new_key(groups, key) {
                    for (existing_key, existing_group_id) in groups.drain(..) {
                        slots[existing_key as usize] = Some(existing_group_id);
                    }
                    slots[key as usize] = Some(group_id);
                    *self = Self::Dense(slots);
                } else {
                    let mut hash = groups.drain(..).collect::<AggregateHashMap<_, _>>();
                    hash.insert(key, group_id);
                    *self = Self::Hash(hash);
                }
            }
            Self::Dense(slots) => {
                let key_index = key as usize;
                if key_index < slots.len() {
                    slots[key_index] = Some(group_id);
                    return;
                }
                let required = key_index.saturating_add(1);
                if required <= dense_u32_group_index_max_slots() {
                    slots.resize(required, None);
                    slots[key_index] = Some(group_id);
                } else {
                    let mut hash = AggregateHashMap::default();
                    for (key, existing_group_id) in slots.iter().enumerate() {
                        if let Some(existing_group_id) = existing_group_id
                            && let Ok(key) = u32::try_from(key)
                        {
                            hash.insert(key, *existing_group_id);
                        }
                    }
                    hash.insert(key, group_id);
                    *self = Self::Hash(hash);
                }
            }
            Self::Hash(groups) => {
                groups.insert(key, group_id);
            }
        }
    }

    fn iter(&self) -> DenseU32GroupIndexIter<'_> {
        match self {
            Self::Small(groups) => DenseU32GroupIndexIter::Small(groups.iter()),
            Self::Dense(slots) => DenseU32GroupIndexIter::Dense { offset: 0, slots },
            Self::Hash(groups) => DenseU32GroupIndexIter::Hash(groups.iter()),
        }
    }
}

fn dense_u32_slots_for_new_key(groups: &[(u32, usize)], key: u32) -> Option<Vec<Option<usize>>> {
    let max = groups
        .iter()
        .fold(key, |max, (candidate, _)| max.max(*candidate));
    let slot_count = usize::try_from(max).ok()?.checked_add(1)?;
    (slot_count <= dense_u32_group_index_max_slots()).then(|| vec![None; slot_count])
}

enum DenseU32GroupIndexIter<'a> {
    Small(std::slice::Iter<'a, (u32, usize)>),
    Dense {
        offset: usize,
        slots: &'a [Option<usize>],
    },
    Hash(std::collections::hash_map::Iter<'a, u32, usize>),
}

impl Iterator for DenseU32GroupIndexIter<'_> {
    type Item = (u32, usize);

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Small(iter) => iter.next().map(|(key, group_id)| (*key, *group_id)),
            Self::Dense { offset, slots } => {
                while *offset < slots.len() {
                    let current = *offset;
                    *offset += 1;
                    if let Some(group_id) = slots[current]
                        && let Ok(key) = u32::try_from(current)
                    {
                        return Some((key, group_id));
                    }
                }
                None
            }
            Self::Hash(iter) => iter.next().map(|(key, group_id)| (*key, *group_id)),
        }
    }
}

struct CoalesceKeyCountSumGroup {
    first: Option<i64>,
    second: Option<i32>,
    third_string_id: Option<u32>,
    count: u64,
    sum: i64,
    sum_count: u64,
}

impl CoalesceKeyCountSumGroup {
    fn new(first: Option<i64>, second: Option<i32>, third_string_id: Option<u32>) -> Self {
        Self {
            first,
            second,
            third_string_id,
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

    fn merge_counts(&mut self, count: u64, sum: i64, sum_count: u64) {
        self.count = self.count.saturating_add(count);
        self.sum = self.sum.saturating_add(sum);
        self.sum_count = self.sum_count.saturating_add(sum_count);
    }

    fn finish(
        self,
        key_len: usize,
        third_strings: &[String],
        aggregates: &[AggregateExpr],
    ) -> GroupAggregateResult {
        let third = self
            .third_string_id
            .map(|id| third_strings[id as usize].clone());
        GroupAggregateResult {
            keys: if key_len == 2 {
                vec![GroupValue::Int64(self.first), GroupValue::Utf8(third)]
            } else {
                vec![
                    GroupValue::Int64(self.first),
                    GroupValue::Date32(self.second),
                    GroupValue::Utf8(third),
                ]
            },
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
    if can_use_single_key_count_sum_path(group_by, aggregates)
        && let Some(groups) = merge_single_key_count_sum_partials(&partials, aggregates)?
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
    } else if can_use_single_key_count_sum_path(group_by, aggregates) {
        metrics.groups = collect_single_key_count_sum_batch_results(&batch, group_by, aggregates)?;
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

pub(crate) struct SingleKeyCountSumVectorState {
    count_expr: AggregateExpr,
    sum_expr: AggregateExpr,
    group_index: CountSumGroupIndex,
    groups: Vec<CountSumGroup>,
}

impl SingleKeyCountSumVectorState {
    pub(crate) fn new_i32(count_expr: AggregateExpr, sum_expr: AggregateExpr) -> Self {
        Self {
            count_expr,
            sum_expr,
            group_index: CountSumGroupIndex::Int32 {
                groups: DenseI32GroupIndex::default(),
                null_group: None,
            },
            groups: Vec::new(),
        }
    }

    pub(crate) fn new_i64(count_expr: AggregateExpr, sum_expr: AggregateExpr) -> Self {
        Self {
            count_expr,
            sum_expr,
            group_index: CountSumGroupIndex::Int64 {
                groups: AdaptiveCopyGroupIndex::default(),
                null_group: None,
            },
            groups: Vec::new(),
        }
    }

    pub(crate) fn consume_i32_i64_batch(&mut self, batch: BatchView<'_>) -> Result<()> {
        let key_values = batch.i32_vector(0).ok_or_else(|| {
            DodamError::UnsupportedSql(
                "direct primitive count/sum projected key is not Int32".to_string(),
            )
        })?;
        let sum_values = batch.i64_vector(1).ok_or_else(|| {
            DodamError::UnsupportedSql(
                "direct primitive count/sum projected sum input is not Int64".to_string(),
            )
        })?;
        if key_values.values_if_null_free().is_none() && key_values.raw_nullable().is_none() {
            return Err(DodamError::UnsupportedSql(
                "direct primitive count/sum requires raw nullable or non-null key".to_string(),
            ));
        }
        let row_count = direct_count_sum_row_count_i32(key_values)?;
        let mut sums = DirectI64SumInput::try_new(sum_values, row_count)?;
        let CountSumGroupIndex::Int32 {
            groups: index,
            null_group,
        } = &mut self.group_index
        else {
            unreachable!("vector count/sum state uses Int32 group index")
        };
        if let Some(keys) = key_values.values_if_null_free() {
            for row in 0..keys.len() {
                let group_id = count_sum_group_id_for_i32(
                    index,
                    keys[row],
                    &mut self.groups,
                    &CountSumValueInput::Int64Raw,
                );
                self.groups[group_id].update_raw_i64_optional(sums.value(row));
            }
        } else if let Some((keys, def_levels)) = key_values.raw_nullable() {
            let mut value_index = 0usize;
            for row in 0..def_levels.len() {
                let group_id = if def_levels[row] == 0 {
                    count_sum_null_group_id(
                        null_group,
                        &mut self.groups,
                        GroupValue::Int64(None),
                        &CountSumValueInput::Int64Raw,
                    )
                } else {
                    let key = keys[value_index];
                    value_index += 1;
                    count_sum_group_id_for_i32(
                        index,
                        key,
                        &mut self.groups,
                        &CountSumValueInput::Int64Raw,
                    )
                };
                self.groups[group_id].update_raw_i64_optional(sums.value(row));
            }
        }
        Ok(())
    }

    pub(crate) fn consume_i64_i64_batch(&mut self, batch: BatchView<'_>) -> Result<()> {
        let key_values = batch.i64_vector(0).ok_or_else(|| {
            DodamError::UnsupportedSql(
                "direct primitive count/sum projected key is not Int64".to_string(),
            )
        })?;
        let sum_values = batch.i64_vector(1).ok_or_else(|| {
            DodamError::UnsupportedSql(
                "direct primitive count/sum projected sum input is not Int64".to_string(),
            )
        })?;
        if key_values.values_if_null_free().is_none() && key_values.raw_nullable().is_none() {
            return Err(DodamError::UnsupportedSql(
                "direct primitive count/sum requires raw nullable or non-null key".to_string(),
            ));
        }
        let row_count = direct_count_sum_row_count_i64(key_values)?;
        let mut sums = DirectI64SumInput::try_new(sum_values, row_count)?;
        let CountSumGroupIndex::Int64 {
            groups: index,
            null_group,
        } = &mut self.group_index
        else {
            unreachable!("vector count/sum state uses Int64 group index")
        };
        if let Some(keys) = key_values.values_if_null_free() {
            for row in 0..keys.len() {
                let group_id = count_sum_group_id_for_i64(
                    index,
                    keys[row],
                    &mut self.groups,
                    &CountSumValueInput::Int64Raw,
                );
                self.groups[group_id].update_raw_i64_optional(sums.value(row));
            }
        } else if let Some((keys, def_levels)) = key_values.raw_nullable() {
            let mut value_index = 0usize;
            for row in 0..def_levels.len() {
                let group_id = if def_levels[row] == 0 {
                    count_sum_null_group_id(
                        null_group,
                        &mut self.groups,
                        GroupValue::Int64(None),
                        &CountSumValueInput::Int64Raw,
                    )
                } else {
                    let key = keys[value_index];
                    value_index += 1;
                    count_sum_group_id_for_i64(
                        index,
                        key,
                        &mut self.groups,
                        &CountSumValueInput::Int64Raw,
                    )
                };
                self.groups[group_id].update_raw_i64_optional(sums.value(row));
            }
        }
        Ok(())
    }

    pub(crate) fn merge(&mut self, partial: Self) -> Result<()> {
        match &mut self.group_index {
            CountSumGroupIndex::Int32 {
                groups: index,
                null_group,
            } => {
                for partial_group in partial.groups {
                    let group_id = match partial_group.key.clone() {
                        GroupValue::Int64(Some(key)) => {
                            let key = i32::try_from(key).map_err(|_| {
                                DodamError::UnsupportedSql(
                                    "direct primitive count/sum partial key out of Int32 range"
                                        .to_string(),
                                )
                            })?;
                            count_sum_group_id_for_i32(
                                index,
                                key,
                                &mut self.groups,
                                &CountSumValueInput::Int64Raw,
                            )
                        }
                        GroupValue::Int64(None) => count_sum_null_group_id(
                            null_group,
                            &mut self.groups,
                            GroupValue::Int64(None),
                            &CountSumValueInput::Int64Raw,
                        ),
                        _ => {
                            return Err(DodamError::UnsupportedSql(
                                "direct primitive count/sum partial key shape mismatch".to_string(),
                            ));
                        }
                    };
                    self.groups[group_id].merge_group(partial_group);
                }
            }
            CountSumGroupIndex::Int64 {
                groups: index,
                null_group,
            } => {
                for partial_group in partial.groups {
                    let group_id = match partial_group.key.clone() {
                        GroupValue::Int64(Some(key)) => count_sum_group_id_for_i64(
                            index,
                            key,
                            &mut self.groups,
                            &CountSumValueInput::Int64Raw,
                        ),
                        GroupValue::Int64(None) => count_sum_null_group_id(
                            null_group,
                            &mut self.groups,
                            GroupValue::Int64(None),
                            &CountSumValueInput::Int64Raw,
                        ),
                        _ => {
                            return Err(DodamError::UnsupportedSql(
                                "direct primitive count/sum partial key shape mismatch".to_string(),
                            ));
                        }
                    };
                    self.groups[group_id].merge_group(partial_group);
                }
            }
            _ => unreachable!("vector count/sum state uses primitive numeric group index"),
        }
        Ok(())
    }

    pub(crate) fn finish(self) -> AggregateMetrics {
        AggregateMetrics {
            groups: finish_count_sum_groups(
                self.group_index,
                self.groups,
                self.count_expr,
                self.sum_expr,
            ),
            ..AggregateMetrics::default()
        }
    }
}

fn direct_count_sum_row_count_i32(key_values: I32VectorView<'_>) -> Result<usize> {
    if let Some(keys) = key_values.values_if_null_free() {
        return Ok(keys.len());
    }
    if let Some((_, def_levels)) = key_values.raw_nullable() {
        return Ok(def_levels.len());
    }
    Err(DodamError::UnsupportedSql(
        "direct primitive count/sum requires raw nullable or non-null key".to_string(),
    ))
}

fn direct_count_sum_row_count_i64(key_values: I64VectorView<'_>) -> Result<usize> {
    if let Some(keys) = key_values.values_if_null_free() {
        return Ok(keys.len());
    }
    if let Some((_, def_levels)) = key_values.raw_nullable() {
        return Ok(def_levels.len());
    }
    Err(DodamError::UnsupportedSql(
        "direct primitive count/sum requires raw nullable or non-null key".to_string(),
    ))
}

enum DirectI64SumInput<'a> {
    NonNull(&'a [i64]),
    RawNullable {
        values: &'a [i64],
        def_levels: &'a [i16],
        value_index: usize,
    },
}

impl<'a> DirectI64SumInput<'a> {
    fn try_new(sum_values: I64VectorView<'a>, row_count: usize) -> Result<Self> {
        if let Some(values) = sum_values.values_if_null_free() {
            if values.len() != row_count {
                return Err(DodamError::UnsupportedSql(
                    "direct primitive count/sum column length mismatch".to_string(),
                ));
            }
            return Ok(Self::NonNull(values));
        }
        if let Some((values, def_levels)) = sum_values.raw_nullable() {
            if def_levels.len() != row_count {
                return Err(DodamError::UnsupportedSql(
                    "direct primitive count/sum column length mismatch".to_string(),
                ));
            }
            return Ok(Self::RawNullable {
                values,
                def_levels,
                value_index: 0,
            });
        }
        Err(DodamError::UnsupportedSql(
            "direct primitive count/sum requires raw nullable or non-null sum input".to_string(),
        ))
    }

    fn value(&mut self, row: usize) -> Option<i64> {
        match self {
            Self::NonNull(values) => Some(values[row]),
            Self::RawNullable {
                values,
                def_levels,
                value_index,
            } => {
                if def_levels[row] == 0 {
                    None
                } else {
                    let value = values[*value_index];
                    *value_index += 1;
                    Some(value)
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct DecimalDateRangeFilter {
    pub(crate) decimal_min: Option<i128>,
    pub(crate) decimal_max: Option<i128>,
    pub(crate) date_min: Option<i32>,
    pub(crate) date_max: Option<i32>,
}

impl DecimalDateRangeFilter {
    pub(crate) fn try_new(
        expr: &Expr,
        decimal_column: &str,
        date_column: &str,
        decimal_scale: i8,
    ) -> Result<Option<Self>> {
        let mut filter = Self::default();
        if !filter.add_expr(expr, decimal_column, date_column, decimal_scale)? {
            return Ok(None);
        }
        Ok(Some(filter))
    }

    fn add_expr(
        &mut self,
        expr: &Expr,
        decimal_column: &str,
        date_column: &str,
        decimal_scale: i8,
    ) -> Result<bool> {
        match expr {
            Expr::Boolean(Some(true)) => Ok(true),
            Expr::And(left, right) => {
                Ok(
                    self.add_expr(left, decimal_column, date_column, decimal_scale)?
                        && self.add_expr(right, decimal_column, date_column, decimal_scale)?,
                )
            }
            Expr::Comparison(comparison) if comparison.column == decimal_column => {
                let Some(value) = literal_to_decimal_raw(&comparison.value, decimal_scale) else {
                    return Ok(false);
                };
                Ok(self.add_decimal_comparison(comparison.op, value))
            }
            Expr::Comparison(comparison) if comparison.column == date_column => {
                let Some(value) = literal_to_date32(&comparison.value) else {
                    return Ok(false);
                };
                Ok(self.add_date_comparison(comparison.op, value))
            }
            _ => Ok(false),
        }
    }

    fn add_decimal_comparison(&mut self, op: ComparisonOp, value: i128) -> bool {
        match op {
            ComparisonOp::Eq => {
                self.decimal_min =
                    Some(self.decimal_min.map_or(value, |current| current.max(value)));
                self.decimal_max =
                    Some(self.decimal_max.map_or(value, |current| current.min(value)));
                true
            }
            ComparisonOp::Lt => {
                let Some(value) = value.checked_sub(1) else {
                    return false;
                };
                self.decimal_max =
                    Some(self.decimal_max.map_or(value, |current| current.min(value)));
                true
            }
            ComparisonOp::LtEq => {
                self.decimal_max =
                    Some(self.decimal_max.map_or(value, |current| current.min(value)));
                true
            }
            ComparisonOp::Gt => {
                let Some(value) = value.checked_add(1) else {
                    return false;
                };
                self.decimal_min =
                    Some(self.decimal_min.map_or(value, |current| current.max(value)));
                true
            }
            ComparisonOp::GtEq => {
                self.decimal_min =
                    Some(self.decimal_min.map_or(value, |current| current.max(value)));
                true
            }
            ComparisonOp::NotEq => false,
        }
    }

    fn add_date_comparison(&mut self, op: ComparisonOp, value: i32) -> bool {
        match op {
            ComparisonOp::Eq => {
                self.date_min = Some(self.date_min.map_or(value, |current| current.max(value)));
                self.date_max = Some(self.date_max.map_or(value, |current| current.min(value)));
                true
            }
            ComparisonOp::Lt => {
                let Some(value) = value.checked_sub(1) else {
                    return false;
                };
                self.date_max = Some(self.date_max.map_or(value, |current| current.min(value)));
                true
            }
            ComparisonOp::LtEq => {
                self.date_max = Some(self.date_max.map_or(value, |current| current.min(value)));
                true
            }
            ComparisonOp::Gt => {
                let Some(value) = value.checked_add(1) else {
                    return false;
                };
                self.date_min = Some(self.date_min.map_or(value, |current| current.max(value)));
                true
            }
            ComparisonOp::GtEq => {
                self.date_min = Some(self.date_min.map_or(value, |current| current.max(value)));
                true
            }
            ComparisonOp::NotEq => false,
        }
    }

    fn matches(&self, decimal: i128, date: i32) -> bool {
        self.decimal_min.is_none_or(|min| decimal >= min)
            && self.decimal_max.is_none_or(|max| decimal <= max)
            && self.date_min.is_none_or(|min| date >= min)
            && self.date_max.is_none_or(|max| date <= max)
    }

    fn matches_i64(&self, decimal: i64, date: i32) -> bool {
        self.matches(i128::from(decimal), date)
    }
}

fn literal_to_decimal_raw(value: &LiteralValue, scale: i8) -> Option<i128> {
    let factor = decimal_scale_factor_i128(scale)?;
    match value {
        LiteralValue::Int64(value) => i128::from(*value).checked_mul(factor),
        LiteralValue::Float64(value) if value.is_finite() => {
            Some((*value * factor as f64).round() as i128)
        }
        LiteralValue::Utf8(value) => parse_decimal_literal_raw(value, scale),
        _ => None,
    }
}

fn decimal_scale_factor_i128(scale: i8) -> Option<i128> {
    let scale = u32::try_from(scale).ok()?;
    10_i128.checked_pow(scale)
}

fn parse_decimal_literal_raw(value: &str, scale: i8) -> Option<i128> {
    if scale < 0 {
        return None;
    }
    let scale = usize::try_from(scale).ok()?;
    let value = value.trim();
    if value.is_empty() {
        return None;
    }
    let negative = value.starts_with('-');
    let unsigned = value.strip_prefix(['-', '+']).unwrap_or(value);
    let (whole, fraction) = unsigned.split_once('.').unwrap_or((unsigned, ""));
    if whole.is_empty() && fraction.is_empty() {
        return None;
    }
    if !whole.bytes().all(|byte| byte.is_ascii_digit())
        || !fraction.bytes().all(|byte| byte.is_ascii_digit())
        || fraction.len() > scale
    {
        return None;
    }
    let mut raw = whole.parse::<i128>().ok()?;
    raw = raw.checked_mul(10_i128.checked_pow(u32::try_from(scale).ok()?)?)?;
    if !fraction.is_empty() {
        let mut fraction_raw = fraction.parse::<i128>().ok()?;
        fraction_raw = fraction_raw
            .checked_mul(10_i128.checked_pow(u32::try_from(scale - fraction.len()).ok()?)?)?;
        raw = raw.checked_add(fraction_raw)?;
    }
    Some(if negative { -raw } else { raw })
}

fn literal_to_date32(value: &LiteralValue) -> Option<i32> {
    match value {
        LiteralValue::Int64(value) => i32::try_from(*value).ok(),
        LiteralValue::Utf8(value) => parse_date32_literal(value),
        _ => None,
    }
}

fn parse_date32_literal(value: &str) -> Option<i32> {
    let mut parts = value.trim().split('-');
    let year = parts.next()?.parse::<i32>().ok()?;
    let month = parts.next()?.parse::<u32>().ok()?;
    let day = parts.next()?.parse::<u32>().ok()?;
    if parts.next().is_some() || !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }
    Some(days_from_civil(year, month, day))
}

fn days_from_civil(year: i32, month: u32, day: u32) -> i32 {
    let year = year - i32::from(month <= 2);
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let yoe = year - era * 400;
    let month = month as i32;
    let day = day as i32;
    let mp = month + if month > 2 { -3 } else { 9 };
    let doy = (153 * mp + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

pub(crate) struct SingleKeyCountSumMinMaxVectorState {
    aggregates: Vec<AggregateExpr>,
    decimal_precision: u8,
    decimal_scale: i8,
    group_index: SingleKeyCountSumMinMaxIndex,
    groups: Vec<SingleKeyCountSumMinMaxGroup>,
}

impl SingleKeyCountSumMinMaxVectorState {
    pub(crate) fn new_i32(
        aggregates: Vec<AggregateExpr>,
        decimal_precision: u8,
        decimal_scale: i8,
    ) -> Self {
        Self {
            aggregates,
            decimal_precision,
            decimal_scale,
            group_index: SingleKeyCountSumMinMaxIndex::Int32 {
                groups: DenseI32GroupIndex::default(),
                null_group: None,
            },
            groups: Vec::new(),
        }
    }

    pub(crate) fn new_i64(
        aggregates: Vec<AggregateExpr>,
        decimal_precision: u8,
        decimal_scale: i8,
    ) -> Self {
        Self {
            aggregates,
            decimal_precision,
            decimal_scale,
            group_index: SingleKeyCountSumMinMaxIndex::Int64 {
                groups: AdaptiveCopyGroupIndex::default(),
                null_group: None,
            },
            groups: Vec::new(),
        }
    }

    pub(crate) fn consume_i32_i64_decimal_date_batch(
        &mut self,
        batch: BatchView<'_>,
        filter: &DecimalDateRangeFilter,
    ) -> Result<()> {
        let key_values = batch.i32_vector(0).ok_or_else(|| {
            DodamError::UnsupportedSql(
                "direct primitive aggregate projected key is not Int32".to_string(),
            )
        })?;
        let sum_values = batch.i64_vector(1).ok_or_else(|| {
            DodamError::UnsupportedSql(
                "direct primitive aggregate projected sum input is not Int64".to_string(),
            )
        })?;
        let decimal_values = batch.decimal128_vector(2).ok_or_else(|| {
            DodamError::UnsupportedSql(
                "direct primitive aggregate projected min input is not Decimal128".to_string(),
            )
        })?;
        let date_values = batch.date32_vector(3).ok_or_else(|| {
            DodamError::UnsupportedSql(
                "direct primitive aggregate projected max input is not Date32".to_string(),
            )
        })?;
        let Some(keys) = key_values.values_if_null_free() else {
            return Err(DodamError::UnsupportedSql(
                "direct primitive aggregate requires non-null key".to_string(),
            ));
        };
        let Some(sums) = sum_values.values_if_null_free() else {
            return Err(DodamError::UnsupportedSql(
                "direct primitive aggregate requires non-null sum input".to_string(),
            ));
        };
        let Some(dates) = date_values.values_if_null_free() else {
            return Err(DodamError::UnsupportedSql(
                "direct primitive aggregate requires non-null date input".to_string(),
            ));
        };
        let SingleKeyCountSumMinMaxIndex::Int32 { groups: index, .. } = &mut self.group_index
        else {
            unreachable!("vector state uses Int32 group index")
        };
        if let Some(decimals) = decimal_values.raw_i64_values() {
            if keys.len() != sums.len() || keys.len() != decimals.len() || keys.len() != dates.len()
            {
                return Err(DodamError::UnsupportedSql(
                    "direct primitive aggregate column length mismatch".to_string(),
                ));
            }
            for row in 0..keys.len() {
                let decimal = decimals[row];
                let date = dates[row];
                if !filter.matches_i64(decimal, date) {
                    continue;
                }
                let group_id = count_sum_min_max_group_id_for_i32(
                    index,
                    keys[row],
                    &mut self.groups,
                    self.decimal_precision,
                    self.decimal_scale,
                );
                self.groups[group_id].update_raw_non_null(sums[row], i128::from(decimal), date);
            }
        } else {
            let decimals = decimal_values.raw_values();
            if keys.len() != sums.len() || keys.len() != decimals.len() || keys.len() != dates.len()
            {
                return Err(DodamError::UnsupportedSql(
                    "direct primitive aggregate column length mismatch".to_string(),
                ));
            }
            for row in 0..keys.len() {
                let decimal = decimals[row];
                let date = dates[row];
                if !filter.matches(decimal, date) {
                    continue;
                }
                let group_id = count_sum_min_max_group_id_for_i32(
                    index,
                    keys[row],
                    &mut self.groups,
                    self.decimal_precision,
                    self.decimal_scale,
                );
                self.groups[group_id].update_raw_non_null(sums[row], decimal, date);
            }
        }
        Ok(())
    }

    pub(crate) fn consume_i64_i64_decimal_date_batch(
        &mut self,
        batch: BatchView<'_>,
        filter: &DecimalDateRangeFilter,
    ) -> Result<()> {
        let key_values = batch.i64_vector(0).ok_or_else(|| {
            DodamError::UnsupportedSql(
                "direct primitive aggregate projected key is not Int64".to_string(),
            )
        })?;
        let sum_values = batch.i64_vector(1).ok_or_else(|| {
            DodamError::UnsupportedSql(
                "direct primitive aggregate projected sum input is not Int64".to_string(),
            )
        })?;
        let decimal_values = batch.decimal128_vector(2).ok_or_else(|| {
            DodamError::UnsupportedSql(
                "direct primitive aggregate projected min input is not Decimal128".to_string(),
            )
        })?;
        let date_values = batch.date32_vector(3).ok_or_else(|| {
            DodamError::UnsupportedSql(
                "direct primitive aggregate projected max input is not Date32".to_string(),
            )
        })?;
        let Some(keys) = key_values.values_if_null_free() else {
            return Err(DodamError::UnsupportedSql(
                "direct primitive aggregate requires non-null key".to_string(),
            ));
        };
        let Some(sums) = sum_values.values_if_null_free() else {
            return Err(DodamError::UnsupportedSql(
                "direct primitive aggregate requires non-null sum input".to_string(),
            ));
        };
        let Some(dates) = date_values.values_if_null_free() else {
            return Err(DodamError::UnsupportedSql(
                "direct primitive aggregate requires non-null date input".to_string(),
            ));
        };
        let SingleKeyCountSumMinMaxIndex::Int64 { groups: index, .. } = &mut self.group_index
        else {
            unreachable!("vector state uses Int64 group index")
        };
        if let Some(decimals) = decimal_values.raw_i64_values() {
            if keys.len() != sums.len() || keys.len() != decimals.len() || keys.len() != dates.len()
            {
                return Err(DodamError::UnsupportedSql(
                    "direct primitive aggregate column length mismatch".to_string(),
                ));
            }
            for row in 0..keys.len() {
                let decimal = decimals[row];
                let date = dates[row];
                if !filter.matches_i64(decimal, date) {
                    continue;
                }
                let group_id = count_sum_min_max_group_id_for_i64(
                    index,
                    keys[row],
                    &mut self.groups,
                    self.decimal_precision,
                    self.decimal_scale,
                );
                self.groups[group_id].update_raw_non_null(sums[row], i128::from(decimal), date);
            }
        } else {
            let decimals = decimal_values.raw_values();
            if keys.len() != sums.len() || keys.len() != decimals.len() || keys.len() != dates.len()
            {
                return Err(DodamError::UnsupportedSql(
                    "direct primitive aggregate column length mismatch".to_string(),
                ));
            }
            for row in 0..keys.len() {
                let decimal = decimals[row];
                let date = dates[row];
                if !filter.matches(decimal, date) {
                    continue;
                }
                let group_id = count_sum_min_max_group_id_for_i64(
                    index,
                    keys[row],
                    &mut self.groups,
                    self.decimal_precision,
                    self.decimal_scale,
                );
                self.groups[group_id].update_raw_non_null(sums[row], decimal, date);
            }
        }
        Ok(())
    }

    pub(crate) fn merge(&mut self, partial: Self) -> Result<()> {
        match &mut self.group_index {
            SingleKeyCountSumMinMaxIndex::Int32 { groups: index, .. } => {
                for partial_group in partial.groups {
                    let GroupValue::Int64(Some(key)) = partial_group.key else {
                        return Err(DodamError::UnsupportedSql(
                            "direct primitive aggregate partial key shape mismatch".to_string(),
                        ));
                    };
                    let key = i32::try_from(key).map_err(|_| {
                        DodamError::UnsupportedSql(
                            "direct primitive aggregate partial key out of Int32 range".to_string(),
                        )
                    })?;
                    let group_id = count_sum_min_max_group_id_for_i32(
                        index,
                        key,
                        &mut self.groups,
                        self.decimal_precision,
                        self.decimal_scale,
                    );
                    self.groups[group_id].merge_group(partial_group);
                }
            }
            SingleKeyCountSumMinMaxIndex::Int64 { groups: index, .. } => {
                for partial_group in partial.groups {
                    let GroupValue::Int64(Some(key)) = partial_group.key else {
                        return Err(DodamError::UnsupportedSql(
                            "direct primitive aggregate partial key shape mismatch".to_string(),
                        ));
                    };
                    let group_id = count_sum_min_max_group_id_for_i64(
                        index,
                        key,
                        &mut self.groups,
                        self.decimal_precision,
                        self.decimal_scale,
                    );
                    self.groups[group_id].merge_group(partial_group);
                }
            }
            _ => unreachable!("vector state uses primitive numeric group index"),
        }
        Ok(())
    }

    pub(crate) fn finish(self) -> AggregateMetrics {
        AggregateMetrics {
            groups: finish_single_key_count_sum_min_max_groups(
                self.group_index,
                self.groups,
                &self.aggregates,
            ),
            ..AggregateMetrics::default()
        }
    }
}

fn merge_single_key_count_sum_partials(
    partials: &[AggregateMetrics],
    aggregates: &[AggregateExpr],
) -> Result<Option<Vec<GroupAggregateResult>>> {
    let mut index = CountSumGroupIndex::Unset;
    let mut groups = Vec::<CountSumGroup>::new();
    for partial in partials {
        for group in &partial.groups {
            let Some(key) = group.keys.first() else {
                return Ok(None);
            };
            let sum_is_float = matches!(
                group.values.get(1).map(|value| &value.value),
                Some(AggregateValue::Float64(_))
            );
            let group_id = match key {
                GroupValue::Utf8(Some(key)) => {
                    index.ensure_type(&DataType::Utf8);
                    let CountSumGroupIndex::Utf8 {
                        groups: key_index, ..
                    } = &mut index
                    else {
                        return Ok(None);
                    };
                    if let Some(group_id) = key_index.get(key).copied() {
                        group_id
                    } else {
                        let group_id = groups.len();
                        key_index.insert(key.clone(), group_id);
                        groups.push(CountSumGroup::new_for_merge(
                            GroupValue::Utf8(Some(key.clone())),
                            sum_is_float,
                        ));
                        group_id
                    }
                }
                GroupValue::Utf8(None) => {
                    index.ensure_type(&DataType::Utf8);
                    let CountSumGroupIndex::Utf8 { null_group, .. } = &mut index else {
                        return Ok(None);
                    };
                    count_sum_null_group_id_for_merge(
                        null_group,
                        &mut groups,
                        GroupValue::Utf8(None),
                        sum_is_float,
                    )
                }
                GroupValue::Int64(Some(key)) => {
                    index.ensure_type(&DataType::Int64);
                    let CountSumGroupIndex::Int64 {
                        groups: key_index, ..
                    } = &mut index
                    else {
                        return Ok(None);
                    };
                    count_sum_group_id_for_i64_merge(key_index, *key, &mut groups, sum_is_float)
                }
                GroupValue::Int64(None) => {
                    index.ensure_type(&DataType::Int64);
                    let CountSumGroupIndex::Int64 { null_group, .. } = &mut index else {
                        return Ok(None);
                    };
                    count_sum_null_group_id_for_merge(
                        null_group,
                        &mut groups,
                        GroupValue::Int64(None),
                        sum_is_float,
                    )
                }
                GroupValue::UInt64(Some(key)) => {
                    index.ensure_type(&DataType::UInt64);
                    let CountSumGroupIndex::UInt64 {
                        groups: key_index, ..
                    } = &mut index
                    else {
                        return Ok(None);
                    };
                    count_sum_group_id_for_u64_merge(key_index, *key, &mut groups, sum_is_float)
                }
                GroupValue::UInt64(None) => {
                    index.ensure_type(&DataType::UInt64);
                    let CountSumGroupIndex::UInt64 { null_group, .. } = &mut index else {
                        return Ok(None);
                    };
                    count_sum_null_group_id_for_merge(
                        null_group,
                        &mut groups,
                        GroupValue::UInt64(None),
                        sum_is_float,
                    )
                }
                _ => return Ok(None),
            };
            groups[group_id].merge_partial_values(&group.values)?;
        }
    }
    Ok(Some(finish_count_sum_groups(
        index,
        groups,
        aggregates[0].clone(),
        aggregates[1].clone(),
    )))
}

pub fn aggregate_metrics_to_batches(
    metrics: &AggregateMetrics,
    group_by: &[String],
    aggregates: &[AggregateExpr],
) -> Result<Vec<RecordBatch>> {
    if group_by.is_empty() {
        return aggregate_values_to_batch(&metrics.values).map(|batch| vec![batch]);
    }
    if let Some(batch) = count_sum_metrics_to_batch(metrics, group_by, aggregates)? {
        return Ok(vec![batch]);
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

fn count_sum_metrics_to_batch(
    metrics: &AggregateMetrics,
    group_by: &[String],
    aggregates: &[AggregateExpr],
) -> Result<Option<RecordBatch>> {
    if !matches!(
        aggregates,
        [
            AggregateExpr::CountStar | AggregateExpr::Count(_),
            AggregateExpr::Sum(_)
        ]
    ) {
        return Ok(None);
    }
    let mut fields = Vec::with_capacity(group_by.len() + 2);
    let mut columns = Vec::with_capacity(group_by.len() + 2);

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

    let mut counts = Vec::with_capacity(metrics.groups.len());
    let mut sums = Vec::with_capacity(metrics.groups.len());
    let mut sum_is_float = false;
    for group in &metrics.groups {
        let Some(count) = group.values.first() else {
            return Ok(None);
        };
        let Some(sum) = group.values.get(1) else {
            return Ok(None);
        };
        match &count.value {
            AggregateValue::Count(value) => counts.push(Some(*value)),
            _ => return Ok(None),
        }
        match &sum.value {
            AggregateValue::Int64(value) => sums.push(*value),
            AggregateValue::Float64(value) => {
                sum_is_float = true;
                sums.push(value.map(|value| value as i64));
            }
            _ => return Ok(None),
        }
    }
    fields.push(Field::new(
        aggregates[0].to_string(),
        DataType::UInt64,
        true,
    ));
    columns.push(Arc::new(UInt64Array::from(counts)));
    if sum_is_float {
        let values = metrics
            .groups
            .iter()
            .map(
                |group| match group.values.get(1).map(|result| &result.value) {
                    Some(AggregateValue::Float64(value)) => *value,
                    _ => None,
                },
            )
            .collect::<Vec<_>>();
        fields.push(Field::new(
            aggregates[1].to_string(),
            DataType::Float64,
            true,
        ));
        columns.push(Arc::new(Float64Array::from(values)));
    } else {
        fields.push(Field::new(aggregates[1].to_string(), DataType::Int64, true));
        columns.push(Arc::new(Int64Array::from(sums)));
    }

    let schema = Arc::new(Schema::new(fields));
    Ok(Some(RecordBatch::try_new(schema, columns)?))
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

        if let Some(dictionary_len) = keys.dictionary_len() {
            let mut dictionary_cache = vec![None; dictionary_len];
            for row in 0..batch.num_rows() {
                let key = keys.key(row);
                let group_id = group_index.group_id_cached(key, &mut groups, &mut dictionary_cache);
                groups[group_id].update(&sum_values, row);
            }
        } else {
            for row in 0..batch.num_rows() {
                let key = keys.key(row);
                let group_id = group_index.group_id(key, &mut groups);
                groups[group_id].update(&sum_values, row);
            }
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
    Int32DateDictionaryUtf8(
        &'a Int32Array,
        &'a Date32Array,
        &'a DictionaryArray<Int32Type>,
        DictionaryStringValues<'a>,
    ),
    Int64DateDictionaryUtf8(
        &'a Int64Array,
        &'a Date32Array,
        &'a DictionaryArray<Int32Type>,
        DictionaryStringValues<'a>,
    ),
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
            (DataType::Int32, DataType::Date32, DataType::Dictionary(key, value))
                if matches!((&**key, &**value), (DataType::Int32, DataType::Utf8)) =>
            {
                let third = third
                    .as_any()
                    .downcast_ref::<DictionaryArray<Int32Type>>()?;
                Some(Self::Int32DateDictionaryUtf8(
                    first.as_any().downcast_ref::<Int32Array>()?,
                    second.as_any().downcast_ref::<Date32Array>()?,
                    third,
                    dictionary_i32_string_values(third)?,
                ))
            }
            (DataType::Int64, DataType::Date32, DataType::Dictionary(key, value))
                if matches!((&**key, &**value), (DataType::Int32, DataType::Utf8)) =>
            {
                let third = third
                    .as_any()
                    .downcast_ref::<DictionaryArray<Int32Type>>()?;
                Some(Self::Int64DateDictionaryUtf8(
                    first.as_any().downcast_ref::<Int64Array>()?,
                    second.as_any().downcast_ref::<Date32Array>()?,
                    third,
                    dictionary_i32_string_values(third)?,
                ))
            }
            _ => None,
        }
    }

    fn key(&self, row: usize) -> ThreeKeyCountSumBorrowedKey<'_> {
        match self {
            Self::Int32DateUtf8(first, second, third) => ThreeKeyCountSumBorrowedKey {
                first: first.is_valid(row).then(|| i64::from(first.value(row))),
                second: second.is_valid(row).then(|| second.value(row)),
                third: Utf8KeyRef::from_option(third.is_valid(row).then(|| third.value(row))),
            },
            Self::Int64DateUtf8(first, second, third) => ThreeKeyCountSumBorrowedKey {
                first: first.is_valid(row).then(|| first.value(row)),
                second: second.is_valid(row).then(|| second.value(row)),
                third: Utf8KeyRef::from_option(third.is_valid(row).then(|| third.value(row))),
            },
            Self::Int32DateDictionaryUtf8(first, second, third, dictionary_values) => {
                ThreeKeyCountSumBorrowedKey {
                    first: first.is_valid(row).then(|| i64::from(first.value(row))),
                    second: second.is_valid(row).then(|| second.value(row)),
                    third: dictionary_utf8_key(third, *dictionary_values, row),
                }
            }
            Self::Int64DateDictionaryUtf8(first, second, third, dictionary_values) => {
                ThreeKeyCountSumBorrowedKey {
                    first: first.is_valid(row).then(|| first.value(row)),
                    second: second.is_valid(row).then(|| second.value(row)),
                    third: dictionary_utf8_key(third, *dictionary_values, row),
                }
            }
        }
    }

    fn dictionary_len(&self) -> Option<usize> {
        match self {
            Self::Int32DateDictionaryUtf8(_, _, _, values)
            | Self::Int64DateDictionaryUtf8(_, _, _, values) => Some(values.len()),
            _ => None,
        }
    }
}

struct ThreeKeyCountSumBorrowedKey<'a> {
    first: Option<i64>,
    second: Option<i32>,
    third: Utf8KeyRef<'a>,
}

#[derive(Clone, Copy)]
enum Utf8KeyRef<'a> {
    Null,
    Str(&'a str),
    Dictionary { id: usize, value: &'a str },
}

impl<'a> Utf8KeyRef<'a> {
    fn from_option(value: Option<&'a str>) -> Self {
        value.map(Self::Str).unwrap_or(Self::Null)
    }

    fn as_option_str(self) -> Option<&'a str> {
        match self {
            Self::Null => None,
            Self::Str(value) | Self::Dictionary { value, .. } => Some(value),
        }
    }

    fn to_owned_string(self) -> Option<String> {
        self.as_option_str().map(str::to_string)
    }
}

fn dictionary_utf8_key<'a>(
    values: &'a DictionaryArray<Int32Type>,
    dictionary_values: DictionaryStringValues<'a>,
    row: usize,
) -> Utf8KeyRef<'a> {
    if values.is_null(row) {
        return Utf8KeyRef::Null;
    }
    let id = values.keys().value(row);
    let Ok(id) = usize::try_from(id) else {
        return Utf8KeyRef::Null;
    };
    let value = std::str::from_utf8(dictionary_values.value_bytes(id))
        .expect("Arrow dictionary Utf8 value should be valid UTF8");
    Utf8KeyRef::Dictionary { id, value }
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
        if let Some(group_id) = third.lookup_key(key.third) {
            return group_id;
        }
        let group_id = groups.len();
        third.insert_key(key.third, group_id);
        groups.push(ThreeKeyCountSumGroup::new(vec![
            GroupValue::Int64(key.first),
            GroupValue::Date32(key.second),
            GroupValue::Utf8(key.third.to_owned_string()),
        ]));
        group_id
    }

    fn group_id_cached(
        &mut self,
        key: ThreeKeyCountSumBorrowedKey<'_>,
        groups: &mut Vec<ThreeKeyCountSumGroup>,
        cache: &mut [Option<usize>],
    ) -> usize {
        let Utf8KeyRef::Dictionary { id, .. } = key.third else {
            return self.group_id(key, groups);
        };
        let third = self
            .first
            .entry(key.first)
            .or_default()
            .second
            .entry(key.second)
            .or_default();
        if let Some(group_id) = cache[id] {
            if third.contains_group(group_id) {
                return group_id;
            }
        }
        if let Some(group_id) = third.lookup_key(key.third) {
            cache[id] = Some(group_id);
            return group_id;
        }
        let group_id = groups.len();
        third.insert_key(key.third, group_id);
        groups.push(ThreeKeyCountSumGroup::new(vec![
            GroupValue::Int64(key.first),
            GroupValue::Date32(key.second),
            GroupValue::Utf8(key.third.to_owned_string()),
        ]));
        cache[id] = Some(group_id);
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
        if let Some(dictionary_len) = key_reader.dictionary_len() {
            let mut dictionary_cache = vec![None; dictionary_len];
            for row in 0..batch.num_rows() {
                let key = key_reader.key(row);
                let group_id = group_index.group_id_cached(
                    key,
                    &mut groups,
                    &sum_input,
                    &mut dictionary_cache,
                )?;
                groups[group_id].update(&sum_input, row);
            }
        } else {
            for row in 0..batch.num_rows() {
                let key = key_reader.key(row);
                let group_id = group_index.group_id(key, &mut groups, &sum_input)?;
                groups[group_id].update(&sum_input, row);
            }
        }
    }

    let mut group_results = groups
        .into_iter()
        .map(|group| group.finish(AggregateExpr::CountStar, sum_expr.clone()))
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
    Int32DictionaryUtf8(
        &'a Int32Array,
        &'a DictionaryArray<Int32Type>,
        DictionaryStringValues<'a>,
    ),
    Int64DictionaryUtf8(
        &'a Int64Array,
        &'a DictionaryArray<Int32Type>,
        DictionaryStringValues<'a>,
    ),
    DictionaryUtf8Int32(
        &'a DictionaryArray<Int32Type>,
        DictionaryStringValues<'a>,
        &'a Int32Array,
    ),
    DictionaryUtf8Int64(
        &'a DictionaryArray<Int32Type>,
        DictionaryStringValues<'a>,
        &'a Int64Array,
    ),
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
            (DataType::Int32, DataType::Dictionary(key, value))
                if matches!((&**key, &**value), (DataType::Int32, DataType::Utf8)) =>
            {
                let second = second
                    .as_any()
                    .downcast_ref::<DictionaryArray<Int32Type>>()?;
                Some(Self::Int32DictionaryUtf8(
                    first.as_any().downcast_ref::<Int32Array>()?,
                    second,
                    dictionary_i32_string_values(second)?,
                ))
            }
            (DataType::Int64, DataType::Dictionary(key, value))
                if matches!((&**key, &**value), (DataType::Int32, DataType::Utf8)) =>
            {
                let second = second
                    .as_any()
                    .downcast_ref::<DictionaryArray<Int32Type>>()?;
                Some(Self::Int64DictionaryUtf8(
                    first.as_any().downcast_ref::<Int64Array>()?,
                    second,
                    dictionary_i32_string_values(second)?,
                ))
            }
            (DataType::Dictionary(key, value), DataType::Int32)
                if matches!((&**key, &**value), (DataType::Int32, DataType::Utf8)) =>
            {
                let first = first
                    .as_any()
                    .downcast_ref::<DictionaryArray<Int32Type>>()?;
                Some(Self::DictionaryUtf8Int32(
                    first,
                    dictionary_i32_string_values(first)?,
                    second.as_any().downcast_ref::<Int32Array>()?,
                ))
            }
            (DataType::Dictionary(key, value), DataType::Int64)
                if matches!((&**key, &**value), (DataType::Int32, DataType::Utf8)) =>
            {
                let first = first
                    .as_any()
                    .downcast_ref::<DictionaryArray<Int32Type>>()?;
                Some(Self::DictionaryUtf8Int64(
                    first,
                    dictionary_i32_string_values(first)?,
                    second.as_any().downcast_ref::<Int64Array>()?,
                ))
            }
            _ => None,
        }
    }

    fn shape(&self) -> TwoKeyCountSumShape {
        match self {
            Self::Int32Utf8(_, _)
            | Self::Int64Utf8(_, _)
            | Self::Int32DictionaryUtf8(_, _, _)
            | Self::Int64DictionaryUtf8(_, _, _) => TwoKeyCountSumShape::IntUtf8,
            Self::Utf8Int32(_, _)
            | Self::Utf8Int64(_, _)
            | Self::DictionaryUtf8Int32(_, _, _)
            | Self::DictionaryUtf8Int64(_, _, _) => TwoKeyCountSumShape::Utf8Int,
        }
    }

    fn key(&self, row: usize) -> TwoKeyCountSumBorrowedKey<'_> {
        match self {
            Self::Int32Utf8(first, second) => TwoKeyCountSumBorrowedKey::IntUtf8(
                first.is_valid(row).then(|| i64::from(first.value(row))),
                Utf8KeyRef::from_option(second.is_valid(row).then(|| second.value(row))),
            ),
            Self::Int64Utf8(first, second) => TwoKeyCountSumBorrowedKey::IntUtf8(
                first.is_valid(row).then(|| first.value(row)),
                Utf8KeyRef::from_option(second.is_valid(row).then(|| second.value(row))),
            ),
            Self::Utf8Int32(first, second) => TwoKeyCountSumBorrowedKey::Utf8Int(
                Utf8KeyRef::from_option(first.is_valid(row).then(|| first.value(row))),
                second.is_valid(row).then(|| i64::from(second.value(row))),
            ),
            Self::Utf8Int64(first, second) => TwoKeyCountSumBorrowedKey::Utf8Int(
                Utf8KeyRef::from_option(first.is_valid(row).then(|| first.value(row))),
                second.is_valid(row).then(|| second.value(row)),
            ),
            Self::Int32DictionaryUtf8(first, second, dictionary_values) => {
                TwoKeyCountSumBorrowedKey::IntUtf8(
                    first.is_valid(row).then(|| i64::from(first.value(row))),
                    dictionary_utf8_key(second, *dictionary_values, row),
                )
            }
            Self::Int64DictionaryUtf8(first, second, dictionary_values) => {
                TwoKeyCountSumBorrowedKey::IntUtf8(
                    first.is_valid(row).then(|| first.value(row)),
                    dictionary_utf8_key(second, *dictionary_values, row),
                )
            }
            Self::DictionaryUtf8Int32(first, dictionary_values, second) => {
                TwoKeyCountSumBorrowedKey::Utf8Int(
                    dictionary_utf8_key(first, *dictionary_values, row),
                    second.is_valid(row).then(|| i64::from(second.value(row))),
                )
            }
            Self::DictionaryUtf8Int64(first, dictionary_values, second) => {
                TwoKeyCountSumBorrowedKey::Utf8Int(
                    dictionary_utf8_key(first, *dictionary_values, row),
                    second.is_valid(row).then(|| second.value(row)),
                )
            }
        }
    }

    fn dictionary_len(&self) -> Option<usize> {
        match self {
            Self::Int32DictionaryUtf8(_, _, values)
            | Self::Int64DictionaryUtf8(_, _, values)
            | Self::DictionaryUtf8Int32(_, values, _)
            | Self::DictionaryUtf8Int64(_, values, _) => Some(values.len()),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TwoKeyCountSumShape {
    IntUtf8,
    Utf8Int,
}

enum TwoKeyCountSumBorrowedKey<'a> {
    IntUtf8(Option<i64>, Utf8KeyRef<'a>),
    Utf8Int(Utf8KeyRef<'a>, Option<i64>),
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
                if let Some(group_id) = second_index.lookup_key(second) {
                    return Ok(group_id);
                }
                let group_id = Self::push_group(
                    groups,
                    vec![
                        GroupValue::Int64(first),
                        GroupValue::Utf8(second.to_owned_string()),
                    ],
                    sum_input,
                );
                second_index.insert_key(second, group_id);
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
                    Utf8KeyRef::Str(first) | Utf8KeyRef::Dictionary { value: first, .. } => {
                        if !first_groups.contains_key(first) {
                            first_groups.insert(first.to_string(), IntSecondGroupIndex::default());
                        }
                        first_groups
                            .get_mut(first)
                            .expect("inserted utf8 first-key group")
                    }
                    Utf8KeyRef::Null => null_first,
                };
                if let Some(group_id) = second_index.lookup(second) {
                    return Ok(group_id);
                }
                let group_id = Self::push_group(
                    groups,
                    vec![
                        GroupValue::Utf8(first.to_owned_string()),
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

    fn group_id_cached(
        &mut self,
        key: TwoKeyCountSumBorrowedKey<'_>,
        groups: &mut Vec<TwoKeyCountSumGroup>,
        sum_input: &CountSumValueInput<'_>,
        cache: &mut [Option<usize>],
    ) -> Result<usize> {
        match key {
            TwoKeyCountSumBorrowedKey::IntUtf8(
                first,
                second @ Utf8KeyRef::Dictionary { id, .. },
            ) => {
                let second_index = match self {
                    Self::IntUtf8(index) => index.entry(first).or_default(),
                    _ => {
                        return Err(DodamError::TypeMismatch(
                            "mixed two-key aggregate key shape mismatch".to_string(),
                        ));
                    }
                };
                if let Some(group_id) = cache[id] {
                    if second_index.contains_group(group_id) {
                        return Ok(group_id);
                    }
                }
                if let Some(group_id) = second_index.lookup_key(second) {
                    cache[id] = Some(group_id);
                    return Ok(group_id);
                }
                let group_id = Self::push_group(
                    groups,
                    vec![
                        GroupValue::Int64(first),
                        GroupValue::Utf8(second.to_owned_string()),
                    ],
                    sum_input,
                );
                second_index.insert_key(second, group_id);
                cache[id] = Some(group_id);
                Ok(group_id)
            }
            TwoKeyCountSumBorrowedKey::Utf8Int(
                first @ Utf8KeyRef::Dictionary { id, value },
                second,
            ) => {
                let second_index = match self {
                    Self::Utf8Int { first_groups, .. } => {
                        if !first_groups.contains_key(value) {
                            first_groups.insert(value.to_string(), IntSecondGroupIndex::default());
                        }
                        first_groups
                            .get_mut(value)
                            .expect("inserted utf8 first-key group")
                    }
                    _ => {
                        return Err(DodamError::TypeMismatch(
                            "mixed two-key aggregate key shape mismatch".to_string(),
                        ));
                    }
                };
                if let Some(group_id) = cache[id] {
                    if second_index.contains_group(group_id) {
                        return Ok(group_id);
                    }
                }
                if let Some(group_id) = second_index.lookup(second) {
                    cache[id] = Some(group_id);
                    return Ok(group_id);
                }
                let group_id = Self::push_group(
                    groups,
                    vec![
                        GroupValue::Utf8(first.to_owned_string()),
                        GroupValue::Int64(second),
                    ],
                    sum_input,
                );
                second_index.insert(second, group_id);
                cache[id] = Some(group_id);
                Ok(group_id)
            }
            key => self.group_id(key, groups, sum_input),
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

    fn lookup_key(&self, value: Utf8KeyRef<'_>) -> Option<usize> {
        self.lookup(value.as_option_str())
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

    fn insert_key(&mut self, value: Utf8KeyRef<'_>, group_id: usize) {
        self.insert(value.as_option_str(), group_id);
    }

    fn contains_group(&self, group_id: usize) -> bool {
        self.null_group == Some(group_id) || self.non_null.values().any(|value| *value == group_id)
    }
}

impl IntSecondGroupIndex {
    fn lookup(&self, value: Option<i64>) -> Option<usize> {
        self.groups.get(&value).copied()
    }

    fn insert(&mut self, value: Option<i64>, group_id: usize) {
        self.groups.insert(value, group_id);
    }

    fn contains_group(&self, group_id: usize) -> bool {
        self.groups.values().any(|value| *value == group_id)
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

    fn finish(self, count_expr: AggregateExpr, sum_expr: AggregateExpr) -> GroupAggregateResult {
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
                    expr: count_expr,
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

        if !update_single_key_count_sum_groups(
            &batch,
            group_by,
            aggregates,
            &mut group_index,
            &mut groups,
        )? {
            return collect_grouped_aggregates_generic(
                stream,
                fragments,
                group_by,
                aggregates,
                Some(batch),
                Some(metrics),
            );
        }
    }

    metrics.groups = finish_count_sum_groups(group_index, groups, aggregates[0].clone(), sum_expr);
    Ok(metrics)
}

fn collect_single_key_count_sum_batch_results(
    batch: &RecordBatch,
    group_by: &[String],
    aggregates: &[AggregateExpr],
) -> Result<Vec<GroupAggregateResult>> {
    let AggregateExpr::Sum(_) = &aggregates[1] else {
        unreachable!("count/sum fast path precondition");
    };
    let mut group_index = CountSumGroupIndex::Unset;
    let mut groups = Vec::<CountSumGroup>::new();
    if !update_single_key_count_sum_groups(
        batch,
        group_by,
        aggregates,
        &mut group_index,
        &mut groups,
    )? {
        let groups = collect_grouped_aggregates_batch(batch.clone(), group_by, aggregates)?;
        return finish_group_map(groups);
    }
    Ok(finish_count_sum_groups(
        group_index,
        groups,
        aggregates[0].clone(),
        aggregates[1].clone(),
    ))
}

pub struct SingleKeyCountSumBatchAccumulator {
    group_by: Vec<String>,
    count_expr: AggregateExpr,
    sum_expr: AggregateExpr,
    group_index: CountSumGroupIndex,
    groups: Vec<CountSumGroup>,
    metrics: AggregateMetrics,
}

impl SingleKeyCountSumBatchAccumulator {
    pub fn try_new(
        fragments: usize,
        group_by: &[String],
        aggregates: &[AggregateExpr],
    ) -> Option<Self> {
        if !can_use_single_key_count_sum_path(group_by, aggregates) {
            return None;
        }
        Some(Self {
            group_by: group_by.to_vec(),
            count_expr: aggregates[0].clone(),
            sum_expr: aggregates[1].clone(),
            group_index: CountSumGroupIndex::Unset,
            groups: Vec::new(),
            metrics: AggregateMetrics {
                fragments,
                ..AggregateMetrics::default()
            },
        })
    }

    pub fn consume_filtered_batch(
        &mut self,
        batch: &RecordBatch,
        mask: &BooleanArray,
    ) -> Result<bool> {
        if batch.num_rows() == 0 {
            return Ok(true);
        }
        self.metrics.batches += 1;
        self.metrics.rows += selected_mask_count(mask);
        update_single_key_count_sum_groups_filtered(
            batch,
            mask,
            &self.group_by,
            &self.sum_expr,
            &mut self.group_index,
            &mut self.groups,
        )
    }

    pub fn finish(mut self) -> AggregateMetrics {
        self.metrics.groups = finish_count_sum_groups(
            self.group_index,
            self.groups,
            self.count_expr,
            self.sum_expr,
        );
        self.metrics
    }
}

fn update_single_key_count_sum_groups(
    batch: &RecordBatch,
    group_by: &[String],
    aggregates: &[AggregateExpr],
    group_index: &mut CountSumGroupIndex,
    groups: &mut Vec<CountSumGroup>,
) -> Result<bool> {
    let AggregateExpr::Sum(sum_column) = &aggregates[1] else {
        unreachable!("count/sum fast path precondition");
    };
    let key_column = batch.column(column_index(batch, &group_by[0])?);
    if !count_sum_key_type_supported(key_column.data_type()) {
        return Ok(false);
    }
    group_index.ensure_type(key_column.data_type());
    let sum_expr = aggregates[1].clone();
    let sum_column = batch.column(column_index(batch, sum_column)?);
    let sum_input = CountSumValueInput::new(sum_column, &sum_expr)?;
    group_index.update_batch(key_column, groups, &sum_input)
}

fn update_single_key_count_sum_groups_filtered(
    batch: &RecordBatch,
    mask: &BooleanArray,
    group_by: &[String],
    sum_expr: &AggregateExpr,
    group_index: &mut CountSumGroupIndex,
    groups: &mut Vec<CountSumGroup>,
) -> Result<bool> {
    let AggregateExpr::Sum(sum_column) = sum_expr else {
        unreachable!("count/sum fast path precondition");
    };
    let key_column = batch.column(column_index(batch, &group_by[0])?);
    if !count_sum_key_type_supported(key_column.data_type()) {
        return Ok(false);
    }
    group_index.ensure_type(key_column.data_type());
    let sum_column = batch.column(column_index(batch, sum_column)?);
    let sum_input = CountSumValueInput::new(sum_column, sum_expr)?;
    group_index.update_batch_filtered(key_column, mask, groups, &sum_input)
}

fn selected_mask_count(mask: &BooleanArray) -> usize {
    (0..mask.len())
        .filter(|row| mask.is_valid(*row) && mask.value(*row))
        .count()
}

enum CountSumValueInput<'a> {
    Int32(&'a Int32Array),
    Int64(&'a Int64Array),
    Float64(&'a Float64Array),
    Int64Raw,
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
        groups: DenseI32GroupIndex,
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

impl CountSumGroupIndex {
    fn ensure_type(&mut self, data_type: &DataType) {
        if !matches!(self, Self::Unset) {
            return;
        }
        *self = match data_type {
            DataType::Utf8 | DataType::Dictionary(_, _) => Self::Utf8 {
                groups: AggregateHashMap::default(),
                null_group: None,
            },
            DataType::Int32 => Self::Int32 {
                groups: DenseI32GroupIndex::default(),
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
            _ => unreachable!("count/sum fast path key type precondition"),
        };
    }

    fn update_batch(
        &mut self,
        key_column: &ArrayRef,
        groups_out: &mut Vec<CountSumGroup>,
        sum_input: &CountSumValueInput<'_>,
    ) -> Result<bool> {
        match self {
            Self::Utf8 { groups, null_group } => {
                if let Some(values) = key_column.as_any().downcast_ref::<StringArray>() {
                    for row in 0..values.len() {
                        let group_id = if values.is_null(row) {
                            count_sum_null_group_id(
                                null_group,
                                groups_out,
                                GroupValue::Utf8(None),
                                sum_input,
                            )
                        } else {
                            count_sum_group_id_for_utf8(
                                groups,
                                values.value(row),
                                groups_out,
                                sum_input,
                            )
                        };
                        groups_out[group_id].update(sum_input, row);
                    }
                    return Ok(true);
                }
                let Some(values) = key_column
                    .as_any()
                    .downcast_ref::<DictionaryArray<Int32Type>>()
                else {
                    return Ok(false);
                };
                update_count_sum_dictionary_utf8_groups(
                    values, groups, null_group, groups_out, sum_input,
                )
            }
            Self::Int32 { groups, null_group } => {
                let values = key_column
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .expect("Int32 group key");
                for row in 0..values.len() {
                    let group_id = if values.is_null(row) {
                        count_sum_null_group_id(
                            null_group,
                            groups_out,
                            GroupValue::Int64(None),
                            sum_input,
                        )
                    } else {
                        count_sum_group_id_for_i32(groups, values.value(row), groups_out, sum_input)
                    };
                    groups_out[group_id].update(sum_input, row);
                }
                Ok(true)
            }
            Self::Int64 { groups, null_group } => {
                let values = key_column
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .expect("Int64 group key");
                for row in 0..values.len() {
                    let group_id = if values.is_null(row) {
                        count_sum_null_group_id(
                            null_group,
                            groups_out,
                            GroupValue::Int64(None),
                            sum_input,
                        )
                    } else {
                        count_sum_group_id_for_i64(groups, values.value(row), groups_out, sum_input)
                    };
                    groups_out[group_id].update(sum_input, row);
                }
                Ok(true)
            }
            Self::UInt64 { groups, null_group } => {
                let values = key_column
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .expect("UInt64 group key");
                for row in 0..values.len() {
                    let group_id = if values.is_null(row) {
                        count_sum_null_group_id(
                            null_group,
                            groups_out,
                            GroupValue::UInt64(None),
                            sum_input,
                        )
                    } else {
                        count_sum_group_id_for_u64(groups, values.value(row), groups_out, sum_input)
                    };
                    groups_out[group_id].update(sum_input, row);
                }
                Ok(true)
            }
            Self::Unset => unreachable!("group index type should be initialized"),
        }
    }

    fn update_batch_filtered(
        &mut self,
        key_column: &ArrayRef,
        mask: &BooleanArray,
        groups_out: &mut Vec<CountSumGroup>,
        sum_input: &CountSumValueInput<'_>,
    ) -> Result<bool> {
        match self {
            Self::Utf8 { groups, null_group } => {
                if let Some(values) = key_column.as_any().downcast_ref::<StringArray>() {
                    for row in 0..values.len() {
                        if !mask.is_valid(row) || !mask.value(row) {
                            continue;
                        }
                        let group_id = if values.is_null(row) {
                            count_sum_null_group_id(
                                null_group,
                                groups_out,
                                GroupValue::Utf8(None),
                                sum_input,
                            )
                        } else {
                            count_sum_group_id_for_utf8(
                                groups,
                                values.value(row),
                                groups_out,
                                sum_input,
                            )
                        };
                        groups_out[group_id].update(sum_input, row);
                    }
                    return Ok(true);
                }
                let Some(values) = key_column
                    .as_any()
                    .downcast_ref::<DictionaryArray<Int32Type>>()
                else {
                    return Ok(false);
                };
                update_count_sum_dictionary_utf8_groups_filtered(
                    values, mask, groups, null_group, groups_out, sum_input,
                )
            }
            Self::Int32 { groups, null_group } => {
                let values = key_column
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .expect("Int32 group key");
                for row in 0..values.len() {
                    if !mask.is_valid(row) || !mask.value(row) {
                        continue;
                    }
                    let group_id = if values.is_null(row) {
                        count_sum_null_group_id(
                            null_group,
                            groups_out,
                            GroupValue::Int64(None),
                            sum_input,
                        )
                    } else {
                        count_sum_group_id_for_i32(groups, values.value(row), groups_out, sum_input)
                    };
                    groups_out[group_id].update(sum_input, row);
                }
                Ok(true)
            }
            Self::Int64 { groups, null_group } => {
                let values = key_column
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .expect("Int64 group key");
                for row in 0..values.len() {
                    if !mask.is_valid(row) || !mask.value(row) {
                        continue;
                    }
                    let group_id = if values.is_null(row) {
                        count_sum_null_group_id(
                            null_group,
                            groups_out,
                            GroupValue::Int64(None),
                            sum_input,
                        )
                    } else {
                        count_sum_group_id_for_i64(groups, values.value(row), groups_out, sum_input)
                    };
                    groups_out[group_id].update(sum_input, row);
                }
                Ok(true)
            }
            Self::UInt64 { groups, null_group } => {
                let values = key_column
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .expect("UInt64 group key");
                for row in 0..values.len() {
                    if !mask.is_valid(row) || !mask.value(row) {
                        continue;
                    }
                    let group_id = if values.is_null(row) {
                        count_sum_null_group_id(
                            null_group,
                            groups_out,
                            GroupValue::UInt64(None),
                            sum_input,
                        )
                    } else {
                        count_sum_group_id_for_u64(groups, values.value(row), groups_out, sum_input)
                    };
                    groups_out[group_id].update(sum_input, row);
                }
                Ok(true)
            }
            Self::Unset => unreachable!("group index type should be initialized"),
        }
    }
}

fn count_sum_key_type_supported(data_type: &DataType) -> bool {
    match data_type {
        DataType::Utf8 | DataType::Int32 | DataType::Int64 | DataType::UInt64 => true,
        DataType::Dictionary(key_type, value_type) => {
            matches!(
                (&**key_type, &**value_type),
                (DataType::Int32, DataType::Utf8)
            )
        }
        _ => false,
    }
}

fn update_count_sum_dictionary_utf8_groups(
    values: &DictionaryArray<Int32Type>,
    groups: &mut AggregateHashMap<String, usize>,
    null_group: &mut Option<usize>,
    groups_out: &mut Vec<CountSumGroup>,
    sum_input: &CountSumValueInput<'_>,
) -> Result<bool> {
    let Some(dictionary_values) = dictionary_i32_string_values(values) else {
        return Ok(false);
    };
    let keys = values.keys().values().as_ref();
    let mut group_ids = vec![None; dictionary_values.len()];
    for row in 0..values.len() {
        let group_id = if values.is_null(row) {
            count_sum_null_group_id(null_group, groups_out, GroupValue::Utf8(None), sum_input)
        } else {
            count_sum_group_id_for_dictionary_utf8_cached(
                groups,
                &dictionary_values,
                keys[row],
                &mut group_ids,
                groups_out,
                sum_input,
            )?
        };
        groups_out[group_id].update(sum_input, row);
    }
    Ok(true)
}

fn update_count_sum_dictionary_utf8_groups_filtered(
    values: &DictionaryArray<Int32Type>,
    mask: &BooleanArray,
    groups: &mut AggregateHashMap<String, usize>,
    null_group: &mut Option<usize>,
    groups_out: &mut Vec<CountSumGroup>,
    sum_input: &CountSumValueInput<'_>,
) -> Result<bool> {
    let Some(dictionary_values) = dictionary_i32_string_values(values) else {
        return Ok(false);
    };
    let keys = values.keys().values().as_ref();
    let mut group_ids = vec![None; dictionary_values.len()];
    for row in 0..values.len() {
        if !mask.is_valid(row) || !mask.value(row) {
            continue;
        }
        let group_id = if values.is_null(row) {
            count_sum_null_group_id(null_group, groups_out, GroupValue::Utf8(None), sum_input)
        } else {
            count_sum_group_id_for_dictionary_utf8_cached(
                groups,
                &dictionary_values,
                keys[row],
                &mut group_ids,
                groups_out,
                sum_input,
            )?
        };
        groups_out[group_id].update(sum_input, row);
    }
    Ok(true)
}

fn count_sum_group_id_for_utf8(
    index: &mut AggregateHashMap<String, usize>,
    key: &str,
    groups: &mut Vec<CountSumGroup>,
    sum_input: &CountSumValueInput<'_>,
) -> usize {
    if let Some(group_id) = index.get(key).copied() {
        return group_id;
    }
    let group_id = groups.len();
    index.insert(key.to_string(), group_id);
    groups.push(CountSumGroup::new(
        GroupValue::Utf8(Some(key.to_string())),
        sum_input,
    ));
    group_id
}

fn count_sum_group_id_for_dictionary_utf8_cached(
    index: &mut AggregateHashMap<String, usize>,
    dictionary_values: &DictionaryStringValues<'_>,
    key: i32,
    cached_group_ids: &mut [Option<usize>],
    groups: &mut Vec<CountSumGroup>,
    sum_input: &CountSumValueInput<'_>,
) -> Result<usize> {
    let key_index = usize::try_from(key).map_err(|_| {
        DodamError::UnsupportedSql("negative dictionary key in Utf8 aggregate".to_string())
    })?;
    if key_index >= dictionary_values.len() {
        return Err(DodamError::UnsupportedSql(
            "dictionary key out of range in Utf8 aggregate".to_string(),
        ));
    }
    if let Some(group_id) = cached_group_ids[key_index] {
        return Ok(group_id);
    }
    let key_bytes = dictionary_values.value_bytes(key_index);
    let key = std::str::from_utf8(key_bytes).map_err(|_| {
        DodamError::UnsupportedSql("invalid UTF8 dictionary value in aggregate".to_string())
    })?;
    let group_id = count_sum_group_id_for_utf8(index, key, groups, sum_input);
    cached_group_ids[key_index] = Some(group_id);
    Ok(group_id)
}

fn count_sum_group_id_for_i32(
    index: &mut DenseI32GroupIndex,
    key: i32,
    groups: &mut Vec<CountSumGroup>,
    sum_input: &CountSumValueInput<'_>,
) -> usize {
    if let Some(group_id) = index.get(key) {
        return group_id;
    }
    let group_id = groups.len();
    index.insert(key, group_id);
    groups.push(CountSumGroup::new(
        GroupValue::Int64(Some(i64::from(key))),
        sum_input,
    ));
    group_id
}

enum DenseI32GroupIndex {
    Small(Vec<(i32, usize)>),
    Dense { min: i32, slots: Vec<Option<usize>> },
    Hash(AggregateHashMap<i32, usize>),
}

impl Default for DenseI32GroupIndex {
    fn default() -> Self {
        Self::Small(Vec::new())
    }
}

impl DenseI32GroupIndex {
    fn get(&self, key: i32) -> Option<usize> {
        match self {
            Self::Small(groups) => groups
                .iter()
                .find_map(|(candidate, group_id)| (*candidate == key).then_some(*group_id)),
            Self::Dense { min, slots } => {
                let offset = usize::try_from(key.checked_sub(*min)?).ok()?;
                slots.get(offset).copied().flatten()
            }
            Self::Hash(groups) => groups.get(&key).copied(),
        }
    }

    fn insert(&mut self, key: i32, group_id: usize) {
        match self {
            Self::Small(groups) if groups.len() < small_group_linear_limit() => {
                groups.push((key, group_id));
            }
            Self::Small(groups) => {
                let (min, max) = min_max_i32_with_new_key(groups, key);
                if let Some(slot_count) = dense_i32_slot_count(min, max) {
                    let mut slots = vec![None; slot_count];
                    for (existing_key, existing_group_id) in groups.drain(..) {
                        slots[(existing_key - min) as usize] = Some(existing_group_id);
                    }
                    slots[(key - min) as usize] = Some(group_id);
                    *self = Self::Dense { min, slots };
                } else {
                    let mut hash = groups.drain(..).collect::<AggregateHashMap<_, _>>();
                    hash.insert(key, group_id);
                    *self = Self::Hash(hash);
                }
            }
            Self::Dense { min, slots } => {
                if let Some(offset) = key
                    .checked_sub(*min)
                    .and_then(|value| usize::try_from(value).ok())
                    && offset < slots.len()
                {
                    slots[offset] = Some(group_id);
                    return;
                }
                if let Some((new_min, new_slots)) = expand_dense_i32_slots(*min, slots, key) {
                    *min = new_min;
                    *slots = new_slots;
                    let offset = usize::try_from(key - *min).expect("dense offset");
                    slots[offset] = Some(group_id);
                } else {
                    let mut hash = AggregateHashMap::default();
                    for (offset, existing_group_id) in slots.iter().enumerate() {
                        if let Some(existing_group_id) = existing_group_id {
                            if let Ok(key) = i32::try_from(i64::from(*min) + offset as i64) {
                                hash.insert(key, *existing_group_id);
                            }
                        }
                    }
                    hash.insert(key, group_id);
                    *self = Self::Hash(hash);
                }
            }
            Self::Hash(groups) => {
                groups.insert(key, group_id);
            }
        }
    }

    fn iter(&self) -> DenseI32GroupIndexIter<'_> {
        match self {
            Self::Small(groups) => DenseI32GroupIndexIter::Small(groups.iter()),
            Self::Dense { min, slots } => DenseI32GroupIndexIter::Dense {
                min: *min,
                offset: 0,
                slots,
            },
            Self::Hash(groups) => DenseI32GroupIndexIter::Hash(groups.iter()),
        }
    }
}

fn min_max_i32_with_new_key(groups: &[(i32, usize)], key: i32) -> (i32, i32) {
    groups
        .iter()
        .fold((key, key), |(min, max), (candidate, _)| {
            (min.min(*candidate), max.max(*candidate))
        })
}

fn dense_i32_slot_count(min: i32, max: i32) -> Option<usize> {
    let span = i64::from(max) - i64::from(min) + 1;
    let slots = usize::try_from(span).ok()?;
    (slots <= dense_i32_group_index_max_slots()).then_some(slots)
}

fn expand_dense_i32_slots(
    min: i32,
    slots: &[Option<usize>],
    key: i32,
) -> Option<(i32, Vec<Option<usize>>)> {
    let old_max = min.checked_add(i32::try_from(slots.len()).ok()?.checked_sub(1)?)?;
    let new_min = min.min(key);
    let new_max = old_max.max(key);
    let slot_count = dense_i32_slot_count(new_min, new_max)?;
    let mut expanded = vec![None; slot_count];
    let old_offset = usize::try_from(i64::from(min) - i64::from(new_min)).ok()?;
    expanded[old_offset..old_offset + slots.len()].copy_from_slice(slots);
    Some((new_min, expanded))
}

enum DenseI32GroupIndexIter<'a> {
    Small(std::slice::Iter<'a, (i32, usize)>),
    Dense {
        min: i32,
        offset: usize,
        slots: &'a [Option<usize>],
    },
    Hash(std::collections::hash_map::Iter<'a, i32, usize>),
}

impl Iterator for DenseI32GroupIndexIter<'_> {
    type Item = (i32, usize);

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Small(iter) => iter.next().map(|(key, group_id)| (*key, *group_id)),
            Self::Dense { min, offset, slots } => {
                while *offset < slots.len() {
                    let current = *offset;
                    *offset += 1;
                    if let Some(group_id) = slots[current] {
                        let key = i32::try_from(i64::from(*min) + current as i64).ok()?;
                        return Some((key, group_id));
                    }
                }
                None
            }
            Self::Hash(iter) => iter.next().map(|(key, group_id)| (*key, *group_id)),
        }
    }
}

fn count_sum_group_id_for_i64(
    index: &mut AdaptiveCopyGroupIndex<i64>,
    key: i64,
    groups: &mut Vec<CountSumGroup>,
    sum_input: &CountSumValueInput<'_>,
) -> usize {
    if let Some(group_id) = index.get(key) {
        return group_id;
    }
    let group_id = groups.len();
    index.insert(key, group_id);
    groups.push(CountSumGroup::new(GroupValue::Int64(Some(key)), sum_input));
    group_id
}

fn count_sum_group_id_for_u64(
    index: &mut AdaptiveCopyGroupIndex<u64>,
    key: u64,
    groups: &mut Vec<CountSumGroup>,
    sum_input: &CountSumValueInput<'_>,
) -> usize {
    if let Some(group_id) = index.get(key) {
        return group_id;
    }
    let group_id = groups.len();
    index.insert(key, group_id);
    groups.push(CountSumGroup::new(GroupValue::UInt64(Some(key)), sum_input));
    group_id
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

fn count_sum_null_group_id_for_merge(
    null_group: &mut Option<usize>,
    groups: &mut Vec<CountSumGroup>,
    key: GroupValue,
    sum_is_float: bool,
) -> usize {
    if let Some(group_id) = *null_group {
        return group_id;
    }
    let group_id = groups.len();
    groups.push(CountSumGroup::new_for_merge(key, sum_is_float));
    *null_group = Some(group_id);
    group_id
}

fn count_sum_group_id_for_i64_merge(
    index: &mut AdaptiveCopyGroupIndex<i64>,
    key: i64,
    groups: &mut Vec<CountSumGroup>,
    sum_is_float: bool,
) -> usize {
    if let Some(group_id) = index.get(key) {
        return group_id;
    }
    let group_id = groups.len();
    index.insert(key, group_id);
    groups.push(CountSumGroup::new_for_merge(
        GroupValue::Int64(Some(key)),
        sum_is_float,
    ));
    group_id
}

fn count_sum_group_id_for_u64_merge(
    index: &mut AdaptiveCopyGroupIndex<u64>,
    key: u64,
    groups: &mut Vec<CountSumGroup>,
    sum_is_float: bool,
) -> usize {
    if let Some(group_id) = index.get(key) {
        return group_id;
    }
    let group_id = groups.len();
    index.insert(key, group_id);
    groups.push(CountSumGroup::new_for_merge(
        GroupValue::UInt64(Some(key)),
        sum_is_float,
    ));
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

    fn new_for_merge(key: GroupValue, sum_is_float: bool) -> Self {
        Self {
            key,
            count: 0,
            sum_i64: 0,
            sum_f64: 0.0,
            sum_is_float,
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

    fn update_raw_i64_optional(&mut self, value: Option<i64>) {
        self.count += 1;
        if let Some(value) = value {
            self.sum_i64 = self.sum_i64.saturating_add(value);
            self.sum_count += 1;
        }
    }

    fn merge_group(&mut self, partial: Self) {
        self.count = self.count.saturating_add(partial.count);
        self.sum_i64 = self.sum_i64.saturating_add(partial.sum_i64);
        self.sum_f64 += partial.sum_f64;
        self.sum_count = self.sum_count.saturating_add(partial.sum_count);
    }

    fn merge_partial_values(&mut self, values: &[AggregateResult]) -> Result<()> {
        let Some(AggregateResult {
            value: AggregateValue::Count(count),
            ..
        }) = values.first()
        else {
            return Ok(());
        };
        self.count = self.count.saturating_add(*count);
        match values.get(1).map(|value| &value.value) {
            Some(AggregateValue::Int64(Some(value))) => {
                self.sum_i64 = self.sum_i64.saturating_add(*value);
                self.sum_count += 1;
            }
            Some(AggregateValue::Float64(Some(value))) => {
                self.sum_f64 += *value;
                self.sum_count += 1;
            }
            Some(AggregateValue::Int64(None) | AggregateValue::Float64(None)) | None => {}
            _ => {
                return Err(DodamError::UnsupportedSql(
                    "partial count/sum aggregate type mismatch".to_string(),
                ));
            }
        }
        Ok(())
    }

    fn finish(self, count_expr: AggregateExpr, sum_expr: AggregateExpr) -> GroupAggregateResult {
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
                    expr: count_expr,
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

fn finish_count_sum_groups(
    group_index: CountSumGroupIndex,
    groups: Vec<CountSumGroup>,
    count_expr: AggregateExpr,
    sum_expr: AggregateExpr,
) -> Vec<GroupAggregateResult> {
    match group_index {
        CountSumGroupIndex::Int32 {
            groups: index,
            null_group,
        } => {
            finish_ordered_count_sum_groups(index.iter(), null_group, groups, count_expr, sum_expr)
        }
        CountSumGroupIndex::Int64 {
            groups: index,
            null_group,
        } => {
            finish_ordered_count_sum_groups(index.iter(), null_group, groups, count_expr, sum_expr)
        }
        CountSumGroupIndex::UInt64 {
            groups: index,
            null_group,
        } => {
            finish_ordered_count_sum_groups(index.iter(), null_group, groups, count_expr, sum_expr)
        }
        _ => {
            let mut group_results = groups
                .into_iter()
                .map(|group| group.finish(count_expr.clone(), sum_expr.clone()))
                .collect::<Vec<_>>();
            group_results.sort_by(|left, right| compare_group_keys(&left.keys, &right.keys));
            group_results
        }
    }
}

fn finish_ordered_count_sum_groups<K, I>(
    index: I,
    null_group: Option<usize>,
    groups: Vec<CountSumGroup>,
    count_expr: AggregateExpr,
    sum_expr: AggregateExpr,
) -> Vec<GroupAggregateResult>
where
    K: Copy + Ord,
    I: Iterator<Item = (K, usize)>,
{
    let mut groups = groups.into_iter().map(Some).collect::<Vec<_>>();
    let mut entries = index.collect::<Vec<_>>();
    entries.sort_by_key(|(key, _)| *key);
    let mut results = Vec::with_capacity(entries.len() + usize::from(null_group.is_some()));
    if let Some(group_id) = null_group
        && let Some(group) = groups.get_mut(group_id).and_then(Option::take)
    {
        results.push(group.finish(count_expr.clone(), sum_expr.clone()));
    }
    for (_, group_id) in entries {
        if let Some(group) = groups.get_mut(group_id).and_then(Option::take) {
            results.push(group.finish(count_expr.clone(), sum_expr.clone()));
        }
    }
    results
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
        groups: DenseI32GroupIndex,
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
                groups: DenseI32GroupIndex::default(),
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
    index: &mut DenseI32GroupIndex,
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

    fn update_raw_non_null(&mut self, sum_value: i64, min_value: i128, max_value: i32) {
        self.count += 1;
        self.sum += sum_value;
        self.sum_count += 1;
        self.min_decimal = Some(match self.min_decimal {
            Some(current) => current.min(min_value),
            None => min_value,
        });
        self.max_date32 = Some(match self.max_date32 {
            Some(current) => current.max(max_value),
            None => max_value,
        });
    }

    fn merge_group(&mut self, partial: Self) {
        self.count = self.count.saturating_add(partial.count);
        self.sum = self.sum.saturating_add(partial.sum);
        self.sum_count = self.sum_count.saturating_add(partial.sum_count);
        if let Some(value) = partial.min_decimal {
            self.min_decimal = Some(match self.min_decimal {
                Some(current) => current.min(value),
                None => value,
            });
        }
        if let Some(value) = partial.max_date32 {
            self.max_date32 = Some(match self.max_date32 {
                Some(current) => current.max(value),
                None => value,
            });
        }
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
