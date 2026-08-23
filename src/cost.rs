use crate::engine::{JoinAlgorithm, JoinExecutionStrategy};
use crate::execution::{JoinBuildSide, JoinType};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JoinCostInput {
    pub left_estimated_bytes: u64,
    pub right_estimated_bytes: u64,
    pub memory_limit_bytes: u64,
    pub requested_algorithm: JoinAlgorithm,
    pub join_type: JoinType,
    pub left_keys: usize,
    pub right_keys: usize,
}

pub fn choose_join_strategy(input: JoinCostInput) -> JoinExecutionStrategy {
    let build_side = choose_build_side(
        input.join_type,
        input.left_estimated_bytes,
        input.right_estimated_bytes,
    );
    let is_inner = input.join_type == JoinType::Inner;
    let is_partitionable = matches!(
        input.join_type,
        JoinType::Inner | JoinType::Full | JoinType::Semi
    );
    let is_single_key_inner = is_inner && input.left_keys == 1 && input.right_keys == 1;

    if input.requested_algorithm == JoinAlgorithm::SortMerge && is_single_key_inner {
        JoinExecutionStrategy::SortMerge
    } else if is_partitionable
        && input.left_estimated_bytes.min(input.right_estimated_bytes) > input.memory_limit_bytes
    {
        JoinExecutionStrategy::PartitionedHash {
            partitions: partition_count(
                input.left_estimated_bytes.min(input.right_estimated_bytes),
                input.memory_limit_bytes,
            ),
            memory_limit_bytes: input.memory_limit_bytes,
        }
    } else {
        JoinExecutionStrategy::Hash { build_side }
    }
}

fn choose_build_side(
    join_type: JoinType,
    left_estimated_bytes: u64,
    right_estimated_bytes: u64,
) -> JoinBuildSide {
    match join_type {
        JoinType::Inner | JoinType::Left | JoinType::Right | JoinType::Full
            if left_estimated_bytes <= right_estimated_bytes =>
        {
            JoinBuildSide::Left
        }
        JoinType::Inner | JoinType::Left | JoinType::Right | JoinType::Full => JoinBuildSide::Right,
        JoinType::Semi => JoinBuildSide::Right,
    }
}

pub fn partition_count(estimated_build_bytes: u64, memory_limit_bytes: u64) -> usize {
    let memory_limit_bytes = memory_limit_bytes.max(1);
    let partitions =
        estimated_build_bytes.saturating_add(memory_limit_bytes - 1) / memory_limit_bytes;
    partitions.clamp(2, 1024) as usize
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PipelineMemoryCostInput {
    pub estimated_rows: u128,
    pub estimated_row_width: u128,
    pub memory_limit_bytes: u128,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineMemoryStrategy {
    InMemory,
    Partitioned { partitions: usize },
    External,
}

pub fn choose_pipeline_memory_strategy(input: PipelineMemoryCostInput) -> PipelineMemoryStrategy {
    let estimated_bytes = input
        .estimated_rows
        .saturating_mul(input.estimated_row_width.max(1));
    let memory_limit = input.memory_limit_bytes.max(1);
    if estimated_bytes <= memory_limit {
        return PipelineMemoryStrategy::InMemory;
    }
    if estimated_bytes <= memory_limit.saturating_mul(1024) {
        return PipelineMemoryStrategy::Partitioned {
            partitions: partition_count_u128(estimated_bytes, memory_limit),
        };
    }
    PipelineMemoryStrategy::External
}

fn partition_count_u128(estimated_bytes: u128, memory_limit_bytes: u128) -> usize {
    let partitions =
        estimated_bytes.saturating_add(memory_limit_bytes - 1) / memory_limit_bytes.max(1);
    partitions.clamp(2, 1024) as usize
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamingLeftDeepJoinCostInput {
    pub table_count: usize,
    pub projected_output_columns: usize,
    pub estimated_final_rows: u128,
}

pub fn choose_streaming_left_deep_join(input: StreamingLeftDeepJoinCostInput) -> bool {
    input.table_count == 3
        && input.projected_output_columns <= 16
        && input.estimated_final_rows <= 10_000_000
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecimalRangeSelectivityInput {
    pub column_min: i128,
    pub column_max: i128,
    pub filter_min: Option<i128>,
    pub filter_max: Option<i128>,
}

impl DecimalRangeSelectivityInput {
    pub fn estimated_selectivity(self) -> Option<f64> {
        let domain_len = self
            .column_max
            .checked_sub(self.column_min)
            .and_then(|value| value.checked_add(1))?;
        if domain_len <= 0 {
            return None;
        }
        let filter_min = self
            .filter_min
            .unwrap_or(self.column_min)
            .max(self.column_min);
        let filter_max = self
            .filter_max
            .unwrap_or(self.column_max)
            .min(self.column_max);
        if filter_min > filter_max {
            return Some(0.0);
        }
        let selected_len = filter_max
            .checked_sub(filter_min)
            .and_then(|value| value.checked_add(1))?;
        Some((selected_len as f64 / domain_len as f64).clamp(0.0, 1.0))
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FusedSelectedAggregateCostInput {
    pub estimated_selectivity: f64,
    pub max_selectivity: f64,
}

pub fn choose_fused_selected_aggregate(input: FusedSelectedAggregateCostInput) -> bool {
    input.estimated_selectivity.is_finite()
        && input.max_selectivity.is_finite()
        && input.estimated_selectivity <= input.max_selectivity.clamp(0.0, 1.0)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkerCostInput {
    pub row_groups: usize,
    pub available_parallelism: usize,
    pub max_workers: usize,
}

pub fn choose_parallel_workers(input: WorkerCostInput) -> usize {
    input
        .available_parallelism
        .max(1)
        .min(input.max_workers.max(1))
        .min(input.row_groups.max(1))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SqlRuleCostInput {
    pub base_rank: u16,
    pub required_features: usize,
    pub matched_features: usize,
    pub required_columns: usize,
    pub matched_required_columns: usize,
    pub estimated_scan_bytes: Option<u64>,
}

pub fn estimate_sql_rule_cost(input: SqlRuleCostInput) -> u32 {
    let mut cost = u32::from(input.base_rank) * 1_000;
    let missing_features = input
        .required_features
        .saturating_sub(input.matched_features) as u32;
    let missing_columns = input
        .required_columns
        .saturating_sub(input.matched_required_columns) as u32;
    cost = cost.saturating_add(input.required_features as u32);
    cost = cost.saturating_add(input.required_columns as u32);
    cost = cost.saturating_add(missing_features * 100);
    cost = cost.saturating_add(missing_columns * 5);
    cost = cost.saturating_sub(input.matched_features as u32 * 10);
    cost = cost.saturating_sub(input.matched_required_columns as u32);
    if let Some(bytes) = input.estimated_scan_bytes {
        cost = cost.saturating_add((bytes / (1024 * 1024)).min(10_000) as u32);
    }
    cost
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SelectedPayloadCostInput {
    pub records: usize,
    pub selected_rows: usize,
    pub selected_runs: usize,
    pub max_selected_ratio: f64,
    pub min_average_run_len: usize,
    pub cached_reread: bool,
    pub cached_min_average_run_len: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectedPayloadDecision {
    Accept,
    EmptySelection,
    SelectedRatio,
    FragmentedRuns,
    RowGroupSpread,
    PayloadColumns,
}

impl SelectedPayloadDecision {
    pub fn accepted(self) -> bool {
        matches!(self, Self::Accept)
    }

    pub fn reason(self) -> &'static str {
        match self {
            Self::Accept => "accepted",
            Self::EmptySelection => "empty-selection",
            Self::SelectedRatio => "selected-ratio",
            Self::FragmentedRuns => "fragmented-runs",
            Self::RowGroupSpread => "row-group-spread",
            Self::PayloadColumns => "payload-columns",
        }
    }
}

pub fn choose_selected_payload(input: SelectedPayloadCostInput) -> SelectedPayloadDecision {
    if input.records == 0 || input.selected_rows == 0 {
        return SelectedPayloadDecision::EmptySelection;
    }
    let selected_ratio = input.selected_rows as f64 / input.records as f64;
    if selected_ratio > input.max_selected_ratio.clamp(0.0, 1.0) {
        return SelectedPayloadDecision::SelectedRatio;
    }
    let min_average_run_len = if input.cached_reread {
        input
            .cached_min_average_run_len
            .min(input.min_average_run_len)
            .max(1)
    } else {
        input.min_average_run_len.max(1)
    };
    let average_run_len = input.selected_rows / input.selected_runs.max(1);
    if average_run_len < min_average_run_len {
        return SelectedPayloadDecision::FragmentedRuns;
    }
    SelectedPayloadDecision::Accept
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SelectedPayloadSpreadCostInput {
    pub selected_rows: usize,
    pub selected_row_groups: usize,
    pub total_row_groups: usize,
    pub missing_payload_columns: usize,
    pub max_selected_row_group_ratio: f64,
    pub max_selected_row_groups: usize,
}

pub fn choose_selected_payload_by_spread(
    input: SelectedPayloadSpreadCostInput,
) -> SelectedPayloadDecision {
    if input.selected_rows == 0 || input.missing_payload_columns == 0 {
        return SelectedPayloadDecision::EmptySelection;
    }
    if input.total_row_groups == 0 {
        return SelectedPayloadDecision::RowGroupSpread;
    }
    let selected_row_groups = input.selected_row_groups.min(input.total_row_groups);
    if selected_row_groups > input.max_selected_row_groups.max(1) {
        return SelectedPayloadDecision::RowGroupSpread;
    }
    let ratio = selected_row_groups as f64 / input.total_row_groups as f64;
    if ratio > input.max_selected_row_group_ratio.clamp(0.0, 1.0) {
        return SelectedPayloadDecision::RowGroupSpread;
    }
    SelectedPayloadDecision::Accept
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LateMaterializationCostInput {
    pub total_rows: usize,
    pub selected_rows: usize,
    pub selector_runs: usize,
    pub max_selected_ratio: Option<f64>,
    pub max_selector_run_ratio: Option<f64>,
    pub max_selector_runs_per_selected: Option<f64>,
    pub io_cost_gate: bool,
    pub predicate_compressed_bytes: u64,
    pub payload_compressed_bytes: u64,
    pub min_io_saving_ratio: f64,
    pub io_override: bool,
}

pub fn late_materialization_selected_ratio(total_rows: usize, selected_rows: usize) -> f64 {
    if total_rows == 0 {
        0.0
    } else {
        selected_rows as f64 / total_rows as f64
    }
}

pub fn late_materialization_estimated_io_saving(
    selected_ratio: f64,
    predicate_compressed_bytes: u64,
    payload_compressed_bytes: u64,
) -> f64 {
    let full = predicate_compressed_bytes.saturating_add(payload_compressed_bytes) as f64;
    if full <= 0.0 {
        return 0.0;
    }
    let late = predicate_compressed_bytes as f64 + payload_compressed_bytes as f64 * selected_ratio;
    ((full - late) / full).clamp(0.0, 1.0)
}

pub fn choose_late_materialization(input: LateMaterializationCostInput) -> bool {
    if input.total_rows == 0 {
        return true;
    }
    if input.max_selected_ratio.is_none()
        && input.max_selector_run_ratio.is_none()
        && input.max_selector_runs_per_selected.is_none()
    {
        return true;
    }
    if let Some(max_selector_run_ratio) = input.max_selector_run_ratio {
        let selector_run_ratio = input.selector_runs as f64 / input.total_rows as f64;
        if selector_run_ratio > max_selector_run_ratio {
            return false;
        }
    }
    if let Some(max_selector_runs_per_selected) = input.max_selector_runs_per_selected {
        if input.selected_rows > 0 {
            let selector_runs_per_selected =
                input.selector_runs as f64 / input.selected_rows as f64;
            if selector_runs_per_selected > max_selector_runs_per_selected {
                return false;
            }
        }
    }

    let selected_ratio = late_materialization_selected_ratio(input.total_rows, input.selected_rows);
    let accepts_selected_ratio = input
        .max_selected_ratio
        .is_none_or(|max_selected_ratio| selected_ratio <= max_selected_ratio);
    if !input.io_cost_gate {
        return accepts_selected_ratio;
    }

    let io_saving = late_materialization_estimated_io_saving(
        selected_ratio,
        input.predicate_compressed_bytes,
        input.payload_compressed_bytes,
    );
    let saves_enough_io = io_saving >= input.min_io_saving_ratio.clamp(0.0, 1.0);
    if accepts_selected_ratio {
        saves_enough_io
    } else {
        input.io_override && saves_enough_io
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProjectionSelectivityCostInput {
    pub predicate_columns: Option<usize>,
    pub payload_columns: Option<usize>,
    pub default_max_selected_ratio: f64,
    pub narrow_payload_cap: f64,
}

pub fn choose_late_materialization_projection_selected_ratio(
    input: ProjectionSelectivityCostInput,
) -> f64 {
    let default_ratio = input.default_max_selected_ratio.clamp(0.0, 1.0);
    let narrow_cap = input.narrow_payload_cap.clamp(0.0, 1.0);
    let Some(predicate_columns) = input.predicate_columns else {
        return default_ratio.min(narrow_cap);
    };
    let Some(payload_columns) = input.payload_columns else {
        return default_ratio.min(narrow_cap);
    };
    if payload_columns >= predicate_columns.saturating_mul(2).max(1) {
        default_ratio
    } else {
        default_ratio.min(narrow_cap)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LateCoalesceGapCostInput {
    pub selected_rows: usize,
    pub natural_selector_runs: usize,
    pub candidate_selector_runs: usize,
    pub candidate_payload_rows: usize,
    pub selector_run_cost_rows: usize,
    pub max_payload_expansion_rows_per_selected: usize,
}

pub fn choose_late_coalesce_candidate(input: LateCoalesceGapCostInput) -> bool {
    if input.selected_rows == 0 {
        return false;
    }
    if input.candidate_payload_rows < input.selected_rows {
        return false;
    }
    let max_payload_rows = input.selected_rows.saturating_add(
        input
            .selected_rows
            .saturating_mul(input.max_payload_expansion_rows_per_selected),
    );
    if input.candidate_payload_rows > max_payload_rows {
        return false;
    }
    let run_cost = input.selector_run_cost_rows.max(1);
    let natural_cost = input
        .selected_rows
        .saturating_add(input.natural_selector_runs.saturating_mul(run_cost));
    let candidate_cost = input
        .candidate_payload_rows
        .saturating_add(input.candidate_selector_runs.saturating_mul(run_cost));
    candidate_cost < natural_cost
}

pub fn choose_late_coalesce_max_gap(
    selected_offsets: &[u32],
    max_gap: usize,
    selector_run_cost_rows: usize,
    max_payload_expansion_rows_per_selected: usize,
) -> usize {
    if max_gap == 0 || selected_offsets.len() <= 1 {
        return 0;
    }
    let mut best_gap = 0usize;
    let mut best_cost = selected_offsets.len().saturating_add(
        selected_offsets
            .len()
            .saturating_mul(selector_run_cost_rows.max(1)),
    );
    let mut candidate = 1usize;
    while candidate <= max_gap {
        let (candidate_runs, candidate_payload_rows) =
            late_coalesce_candidate_metrics(selected_offsets, candidate);
        if choose_late_coalesce_candidate(LateCoalesceGapCostInput {
            selected_rows: selected_offsets.len(),
            natural_selector_runs: selected_offsets.len(),
            candidate_selector_runs: candidate_runs,
            candidate_payload_rows,
            selector_run_cost_rows,
            max_payload_expansion_rows_per_selected,
        }) {
            let candidate_cost = candidate_payload_rows
                .saturating_add(candidate_runs.saturating_mul(selector_run_cost_rows.max(1)));
            if candidate_cost < best_cost {
                best_cost = candidate_cost;
                best_gap = candidate;
            }
        }
        candidate = candidate.saturating_mul(2);
        if candidate == 0 {
            break;
        }
    }
    if best_gap < max_gap {
        let (candidate_runs, candidate_payload_rows) =
            late_coalesce_candidate_metrics(selected_offsets, max_gap);
        if choose_late_coalesce_candidate(LateCoalesceGapCostInput {
            selected_rows: selected_offsets.len(),
            natural_selector_runs: selected_offsets.len(),
            candidate_selector_runs: candidate_runs,
            candidate_payload_rows,
            selector_run_cost_rows,
            max_payload_expansion_rows_per_selected,
        }) {
            let candidate_cost = candidate_payload_rows
                .saturating_add(candidate_runs.saturating_mul(selector_run_cost_rows.max(1)));
            if candidate_cost < best_cost {
                best_gap = max_gap;
            }
        }
    }
    best_gap
}

fn late_coalesce_candidate_metrics(selected_offsets: &[u32], max_gap: usize) -> (usize, usize) {
    if selected_offsets.is_empty() {
        return (0, 0);
    }
    let mut runs = 1usize;
    let mut payload_rows = 1usize;
    let mut previous = selected_offsets[0] as usize;
    for &offset in &selected_offsets[1..] {
        let offset = offset as usize;
        if offset <= previous {
            continue;
        }
        let gap = offset - previous - 1;
        if gap <= max_gap {
            payload_rows = payload_rows.saturating_add(gap).saturating_add(1);
        } else {
            runs = runs.saturating_add(1);
            payload_rows = payload_rows.saturating_add(1);
        }
        previous = offset;
    }
    (runs, payload_rows)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpressionAggregateLateChunkCostInput {
    pub ordered_output: bool,
    pub output_limit: Option<usize>,
    pub default_chunk: usize,
    pub ordered_limit_chunk: usize,
}

pub fn choose_expression_aggregate_late_row_group_chunk(
    input: ExpressionAggregateLateChunkCostInput,
) -> usize {
    let default_chunk = input.default_chunk.max(1);
    if input.ordered_output && input.output_limit.is_some() {
        input.ordered_limit_chunk.max(default_chunk)
    } else {
        default_chunk
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowGroupMapChunkCostInput {
    pub requested_chunk: Option<usize>,
    pub default_chunk: usize,
}

pub fn choose_row_group_map_chunk(input: RowGroupMapChunkCostInput) -> usize {
    input
        .requested_chunk
        .filter(|chunk| *chunk > 0)
        .unwrap_or_else(|| input.default_chunk.max(1))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrimitiveParquetSinkCostInput {
    pub supported: bool,
    pub rows: usize,
    pub min_rows: usize,
}

pub fn choose_primitive_parquet_sink(input: PrimitiveParquetSinkCostInput) -> bool {
    input.supported && input.rows >= input.min_rows.max(1)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OrderedPrimitiveSinkCostInput {
    pub limit: usize,
    pub small_limit_rows: usize,
    pub large_auto_enabled: bool,
}

pub fn choose_ordered_primitive_sink(input: OrderedPrimitiveSinkCostInput) -> bool {
    input.limit <= input.small_limit_rows.max(1)
        || (input.large_auto_enabled && input.limit > input.small_limit_rows.max(1))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OrderedPrimitiveChunkCostInput {
    pub limit: usize,
    pub row_groups: usize,
    pub available_parallelism: usize,
    pub small_limit_rows: usize,
    pub max_workers: usize,
}

pub fn choose_ordered_primitive_row_group_chunk(input: OrderedPrimitiveChunkCostInput) -> usize {
    if input.limit <= input.small_limit_rows.max(1) {
        return 1;
    }
    choose_parallel_workers(WorkerCostInput {
        row_groups: input.row_groups,
        available_parallelism: input.available_parallelism,
        max_workers: input.max_workers,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimitiveOrderLimitStrategy {
    OrderedScan,
    PostScanTopK,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrimitiveOrderLimitCostInput {
    pub has_limit: bool,
    pub offset: usize,
    pub sort_keys: usize,
    pub descending: bool,
    pub nulls_first: bool,
    pub sort_key_projected: bool,
    pub sort_key_is_i64: bool,
}

pub fn choose_primitive_order_limit_strategy(
    input: PrimitiveOrderLimitCostInput,
) -> PrimitiveOrderLimitStrategy {
    if input.has_limit
        && input.offset == 0
        && input.sort_keys == 1
        && input.descending
        && !input.nulls_first
        && input.sort_key_projected
        && input.sort_key_is_i64
    {
        PrimitiveOrderLimitStrategy::PostScanTopK
    } else {
        PrimitiveOrderLimitStrategy::OrderedScan
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DenseI32AggregateCostInput {
    pub rows: usize,
    pub min_rows: usize,
    pub sample_rows: usize,
    pub sample_unique: usize,
    pub max_sample_unique: usize,
}

pub fn choose_dense_i32_block_accumulate(input: DenseI32AggregateCostInput) -> bool {
    input.sample_rows >= 64
        && input.rows >= input.min_rows.max(1)
        && input.sample_unique <= input.max_sample_unique.max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decimal_range_selectivity_estimates_filter_overlap() {
        let selectivity = DecimalRangeSelectivityInput {
            column_min: 0,
            column_max: 9999,
            filter_min: None,
            filter_max: Some(99),
        }
        .estimated_selectivity()
        .expect("selectivity");
        assert!((selectivity - 0.01).abs() < 0.000001);
    }

    #[test]
    fn decimal_range_selectivity_handles_empty_overlap() {
        let selectivity = DecimalRangeSelectivityInput {
            column_min: 0,
            column_max: 9999,
            filter_min: Some(10_000),
            filter_max: Some(20_000),
        }
        .estimated_selectivity()
        .expect("selectivity");
        assert_eq!(selectivity, 0.0);
    }

    #[test]
    fn selected_payload_decision_reports_reason() {
        assert_eq!(
            choose_selected_payload(SelectedPayloadCostInput {
                records: 10_000,
                selected_rows: 100,
                selected_runs: 2,
                max_selected_ratio: 0.2,
                min_average_run_len: 32,
                cached_reread: false,
                cached_min_average_run_len: 8,
            }),
            SelectedPayloadDecision::Accept
        );
        assert_eq!(
            choose_selected_payload(SelectedPayloadCostInput {
                records: 10_000,
                selected_rows: 5_000,
                selected_runs: 2,
                max_selected_ratio: 0.2,
                min_average_run_len: 32,
                cached_reread: false,
                cached_min_average_run_len: 8,
            }),
            SelectedPayloadDecision::SelectedRatio
        );
        assert_eq!(
            choose_selected_payload(SelectedPayloadCostInput {
                records: 10_000,
                selected_rows: 100,
                selected_runs: 100,
                max_selected_ratio: 0.2,
                min_average_run_len: 32,
                cached_reread: false,
                cached_min_average_run_len: 8,
            }),
            SelectedPayloadDecision::FragmentedRuns
        );
        assert_eq!(
            choose_selected_payload(SelectedPayloadCostInput {
                records: 10_000,
                selected_rows: 800,
                selected_runs: 100,
                max_selected_ratio: 0.2,
                min_average_run_len: 32,
                cached_reread: true,
                cached_min_average_run_len: 8,
            }),
            SelectedPayloadDecision::Accept
        );
    }

    #[test]
    fn selected_payload_spread_rejects_wide_second_pass() {
        assert_eq!(
            choose_selected_payload_by_spread(SelectedPayloadSpreadCostInput {
                selected_rows: 5_000,
                selected_row_groups: 49,
                total_row_groups: 49,
                missing_payload_columns: 1,
                max_selected_row_group_ratio: 0.25,
                max_selected_row_groups: 16,
            }),
            SelectedPayloadDecision::RowGroupSpread
        );
        assert_eq!(
            choose_selected_payload_by_spread(SelectedPayloadSpreadCostInput {
                selected_rows: 5_000,
                selected_row_groups: 4,
                total_row_groups: 49,
                missing_payload_columns: 1,
                max_selected_row_group_ratio: 0.25,
                max_selected_row_groups: 16,
            }),
            SelectedPayloadDecision::Accept
        );
        assert_eq!(
            choose_selected_payload_by_spread(SelectedPayloadSpreadCostInput {
                selected_rows: 5_000,
                selected_row_groups: 4,
                total_row_groups: 49,
                missing_payload_columns: 0,
                max_selected_row_group_ratio: 0.25,
                max_selected_row_groups: 16,
            }),
            SelectedPayloadDecision::EmptySelection
        );
    }

    #[test]
    fn parallel_workers_are_bounded() {
        assert_eq!(
            choose_parallel_workers(WorkerCostInput {
                row_groups: 49,
                available_parallelism: 32,
                max_workers: 12,
            }),
            12
        );
        assert_eq!(
            choose_parallel_workers(WorkerCostInput {
                row_groups: 4,
                available_parallelism: 32,
                max_workers: 12,
            }),
            4
        );
    }

    #[test]
    fn late_materialization_accepts_io_saving_override() {
        assert!(choose_late_materialization(LateMaterializationCostInput {
            total_rows: 1_000,
            selected_rows: 600,
            selector_runs: 10,
            max_selected_ratio: Some(0.1),
            max_selector_run_ratio: Some(0.1),
            max_selector_runs_per_selected: None,
            io_cost_gate: true,
            predicate_compressed_bytes: 1,
            payload_compressed_bytes: 100,
            min_io_saving_ratio: 0.2,
            io_override: true,
        }));
        assert!(!choose_late_materialization(LateMaterializationCostInput {
            total_rows: 1_000,
            selected_rows: 600,
            selector_runs: 200,
            max_selected_ratio: Some(0.1),
            max_selector_run_ratio: Some(0.1),
            max_selector_runs_per_selected: None,
            io_cost_gate: true,
            predicate_compressed_bytes: 1,
            payload_compressed_bytes: 100,
            min_io_saving_ratio: 0.2,
            io_override: true,
        }));
    }

    #[test]
    fn late_materialization_rejects_fragmented_selected_payload() {
        assert!(!choose_late_materialization(LateMaterializationCostInput {
            total_rows: 1_000,
            selected_rows: 100,
            selector_runs: 201,
            max_selected_ratio: Some(0.2),
            max_selector_run_ratio: Some(0.5),
            max_selector_runs_per_selected: Some(2.0),
            io_cost_gate: false,
            predicate_compressed_bytes: 1,
            payload_compressed_bytes: 100,
            min_io_saving_ratio: 0.2,
            io_override: true,
        }));
        assert!(choose_late_materialization(LateMaterializationCostInput {
            total_rows: 1_000,
            selected_rows: 100,
            selector_runs: 200,
            max_selected_ratio: Some(0.2),
            max_selector_run_ratio: Some(0.5),
            max_selector_runs_per_selected: Some(2.0),
            io_cost_gate: false,
            predicate_compressed_bytes: 1,
            payload_compressed_bytes: 100,
            min_io_saving_ratio: 0.2,
            io_override: true,
        }));
    }

    #[test]
    fn projection_selected_ratio_caps_narrow_payload() {
        assert_eq!(
            choose_late_materialization_projection_selected_ratio(ProjectionSelectivityCostInput {
                predicate_columns: Some(2),
                payload_columns: Some(1),
                default_max_selected_ratio: 0.75,
                narrow_payload_cap: 0.35,
            }),
            0.35
        );
        assert_eq!(
            choose_late_materialization_projection_selected_ratio(ProjectionSelectivityCostInput {
                predicate_columns: Some(1),
                payload_columns: Some(2),
                default_max_selected_ratio: 0.75,
                narrow_payload_cap: 0.35,
            }),
            0.75
        );
    }

    #[test]
    fn late_coalesce_gap_balances_runs_against_payload_rows() {
        assert_eq!(choose_late_coalesce_max_gap(&[1, 3, 5, 100], 8, 4, 8), 1);
        assert_eq!(choose_late_coalesce_max_gap(&[1, 20, 40], 8, 4, 8), 0);
        assert_eq!(choose_late_coalesce_max_gap(&[1, 3, 5, 7], 8, 4, 0), 0);
        assert_eq!(choose_late_coalesce_max_gap(&[1, 3, 5, 7], 8, 4, 8), 1);
    }

    #[test]
    fn primitive_sink_and_dense_gates_are_data_driven() {
        assert!(choose_primitive_parquet_sink(
            PrimitiveParquetSinkCostInput {
                supported: true,
                rows: 65_536,
                min_rows: 64 * 1024,
            }
        ));
        assert!(choose_ordered_primitive_sink(
            OrderedPrimitiveSinkCostInput {
                limit: 1_000_000,
                small_limit_rows: 16 * 1024,
                large_auto_enabled: true,
            }
        ));
        assert!(choose_dense_i32_block_accumulate(
            DenseI32AggregateCostInput {
                rows: 256,
                min_rows: 1,
                sample_rows: 256,
                sample_unique: 8,
                max_sample_unique: 32,
            }
        ));
    }

    #[test]
    fn primitive_order_limit_strategy_prefers_post_scan_topk_only_for_safe_desc_limit() {
        assert_eq!(
            choose_primitive_order_limit_strategy(PrimitiveOrderLimitCostInput {
                has_limit: true,
                offset: 0,
                sort_keys: 1,
                descending: true,
                nulls_first: false,
                sort_key_projected: true,
                sort_key_is_i64: true,
            }),
            PrimitiveOrderLimitStrategy::PostScanTopK
        );
        assert_eq!(
            choose_primitive_order_limit_strategy(PrimitiveOrderLimitCostInput {
                has_limit: true,
                offset: 10,
                sort_keys: 1,
                descending: true,
                nulls_first: false,
                sort_key_projected: true,
                sort_key_is_i64: true,
            }),
            PrimitiveOrderLimitStrategy::OrderedScan
        );
        assert_eq!(
            choose_primitive_order_limit_strategy(PrimitiveOrderLimitCostInput {
                has_limit: true,
                offset: 0,
                sort_keys: 1,
                descending: false,
                nulls_first: false,
                sort_key_projected: true,
                sort_key_is_i64: true,
            }),
            PrimitiveOrderLimitStrategy::OrderedScan
        );
    }

    #[test]
    fn pipeline_memory_strategy_escalates_from_memory_to_external() {
        assert_eq!(
            choose_pipeline_memory_strategy(PipelineMemoryCostInput {
                estimated_rows: 1_000,
                estimated_row_width: 64,
                memory_limit_bytes: 128 * 1024,
            }),
            PipelineMemoryStrategy::InMemory
        );
        assert_eq!(
            choose_pipeline_memory_strategy(PipelineMemoryCostInput {
                estimated_rows: 10_000,
                estimated_row_width: 64,
                memory_limit_bytes: 128 * 1024,
            }),
            PipelineMemoryStrategy::Partitioned { partitions: 5 }
        );
        assert_eq!(
            choose_pipeline_memory_strategy(PipelineMemoryCostInput {
                estimated_rows: 1_000_000_000,
                estimated_row_width: 1024,
                memory_limit_bytes: 128 * 1024,
            }),
            PipelineMemoryStrategy::External
        );
    }

    #[test]
    fn streaming_left_deep_join_rule_is_conservative() {
        assert!(choose_streaming_left_deep_join(
            StreamingLeftDeepJoinCostInput {
                table_count: 3,
                projected_output_columns: 8,
                estimated_final_rows: 1_000_000,
            }
        ));
        assert!(!choose_streaming_left_deep_join(
            StreamingLeftDeepJoinCostInput {
                table_count: 4,
                projected_output_columns: 8,
                estimated_final_rows: 1_000_000,
            }
        ));
        assert!(!choose_streaming_left_deep_join(
            StreamingLeftDeepJoinCostInput {
                table_count: 3,
                projected_output_columns: 64,
                estimated_final_rows: 1_000_000,
            }
        ));
    }

    #[test]
    fn expression_aggregate_late_chunk_grows_for_ordered_limit() {
        assert_eq!(
            choose_expression_aggregate_late_row_group_chunk(
                ExpressionAggregateLateChunkCostInput {
                    ordered_output: true,
                    output_limit: Some(2_000),
                    default_chunk: 2,
                    ordered_limit_chunk: 4,
                }
            ),
            4
        );
        assert_eq!(
            choose_expression_aggregate_late_row_group_chunk(
                ExpressionAggregateLateChunkCostInput {
                    ordered_output: false,
                    output_limit: Some(2_000),
                    default_chunk: 2,
                    ordered_limit_chunk: 4,
                }
            ),
            2
        );
    }

    #[test]
    fn row_group_map_chunk_uses_requested_or_sanitized_default() {
        assert_eq!(
            choose_row_group_map_chunk(RowGroupMapChunkCostInput {
                requested_chunk: Some(8),
                default_chunk: 2,
            }),
            8
        );
        assert_eq!(
            choose_row_group_map_chunk(RowGroupMapChunkCostInput {
                requested_chunk: Some(0),
                default_chunk: 2,
            }),
            2
        );
        assert_eq!(
            choose_row_group_map_chunk(RowGroupMapChunkCostInput {
                requested_chunk: None,
                default_chunk: 0,
            }),
            1
        );
    }
}
