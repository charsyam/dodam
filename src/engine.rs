use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, mpsc};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use arrow::array::{Array, Date32Array, Decimal128Array, Int64Array, UInt32Array};
use arrow::datatypes::DataType;
use arrow::ipc::writer::FileWriter as IpcFileWriter;
use arrow::record_batch::RecordBatch;
use arrow_row::{RowConverter, SortField};
use arrow_select::take::take_record_batch;
use parquet::arrow::arrow_reader::{ParquetRecordBatchReaderBuilder, RowSelection, RowSelector};

use crate::catalog::{
    FileFragmentStatistics, LocalParquetTable, PersistentCatalog, StorageFormat, TableProvider,
    TableScanSource, TableStatistics,
};
use crate::cost::{
    JoinCostInput, LateMaterializationCostInput, RowGroupMapChunkCostInput, WorkerCostInput,
    choose_join_strategy, choose_late_coalesce_max_gap, choose_late_materialization,
    choose_parallel_workers, choose_row_group_map_chunk,
};
use crate::dense::DenseI64BoolLookup;
use crate::error::{DodamError, Result};
use crate::execution::metrics::ScanPlanMetricsCounter;
use crate::execution::{
    AggregateExpr, AggregateMetrics, ComparisonExpr, ComparisonOp, CountSumMinMaxMaxKind,
    DecimalDateRangeFilter, DirectPrimitiveFoldExec, DistinctExec, Expr, FilterExec, FilterExpr,
    FinalMergeExec, GroupAggregateResult, HashJoinExec, IpcExec, JoinBuildSide, JoinType,
    LimitExec, LiteralValue, LocalFoldExec, MemoryExec, PartitionedHashJoinExec,
    PartitionedHashJoinOptions, PhysicalPlan, PredicateSet, Projection, ProjectionExec,
    RecordBatchSink, ScanExec, ScanMetrics, ScanPlanMetrics, SendableBatchStream,
    SingleKeyCountSumBatchAccumulator, SingleKeyCountSumMinMaxVectorState, SortExec, SortExpr,
    SortKey, SortMergeJoinExec, can_merge_partial_aggregates, collect_aggregates,
    collect_grouped_aggregates, collect_metrics, evaluate_filter_mask,
    merge_partial_aggregate_metrics, scan_projection, write_stream_to_sink,
};
use crate::optimizer::LogicalOptimizer;
use crate::plan::{
    DirectPrimitiveFoldMode, ExchangeKind, ExecutionGraphPlan, LogicalPlan, LogicalScan,
    PhysicalExecutionConfig, PhysicalJoinStrategy, PhysicalOperator, PhysicalPlanNode,
    PlanTableSource, TaskInput, TaskPlan,
};
use crate::storage::{
    DirectColumnScanMetrics, DirectI32I32DictionaryI64SelectedBatch,
    DirectI32I64DecimalI32SelectedBatch, DirectOrderedPrimitiveBatch,
    DirectPrimitiveColumnScanMetrics, DirectPrimitiveColumnSpec, DirectPrimitiveColumnType,
    DirectSelectedPrimitivePageBatch, I64BloomPredicate, LocalFileSystemObjectStore, ObjectStore,
    ParquetBatchReader, ParquetFileCache, ParquetFileCacheStats, ParquetMetadataCache,
    PrimitiveRowGroupMinMax,
    collect_parquet_i32_predicate_i64_lookup_selected_i64_mapped_with_store,
    collect_parquet_i64_by_utf8_dictionary_predicate_with_store,
    collect_parquet_i64_two_utf8_i64_mapped_with_store, parquet_column_monotonic_by_scan,
    parquet_row_group_count_with_store, parquet_row_groups_monotonic_by_column,
    parquet_total_row_count_with_store, plan_parquet_scan_tasks, read_parquet_file_statistics,
    read_parquet_i64_column_constant, read_parquet_i64_column_max,
    read_parquet_i128_column_min_max, read_parquet_i128_column_min_max_relaxed,
    read_parquet_primitive_column_min_max_by_row_group, read_parquet_projection_compressed_bytes,
    scan_parquet_dictionary_date_range_selected_primitive_columns_with_store,
    scan_parquet_i32_byte_array_columns_with_store,
    scan_parquet_i32_byte_array_selected_by_i32_with_store,
    scan_parquet_i32_dictionary_id_columns_raw_with_store,
    scan_parquet_i32_dictionary_id_columns_with_store, scan_parquet_i32_i32_columns_with_store,
    scan_parquet_i32_i32_dictionary_i64_decimal_selected_typed_with_store,
    scan_parquet_i32_i64_byte_array_columns_with_store,
    scan_parquet_i32_i64_decimal_i32_selected_typed_with_store,
    scan_parquet_i32_i64_decimal_i32_selected_with_store,
    scan_parquet_i32_i64_dictionary_id_columns_with_store,
    scan_parquet_i32_selected_by_byte_array_prefix_with_store,
    scan_parquet_i64_i64_selected_raw_columns_with_store,
    scan_parquet_i64_lookup_decimal_utf8_selected_primitive_columns_with_store,
    scan_parquet_i64_lookup_staged_two_i64_selected_primitive_columns_with_store,
    scan_parquet_primitive_columns_with_store,
    scan_parquet_primitive_columns_with_store_page_reader,
    scan_parquet_required_plain_primitive_in_list_desc_selected_pages_with_store,
    scan_parquet_required_plain_primitive_in_list_desc_with_store,
};
use crate::vector::{BatchView, Date32VectorView, Decimal128VectorView, I64VectorView};

const LOCAL_SHUFFLE_FILE_TARGET_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Clone)]
pub struct DodamEngine {
    metadata_cache: Arc<ParquetMetadataCache>,
    file_cache: Arc<ParquetFileCache>,
    i128_column_min_max_cache: Arc<Mutex<HashMap<I128ColumnMinMaxCacheKey, Option<(i128, i128)>>>>,
    monotonic_column_scan_cache: Arc<Mutex<HashMap<MonotonicColumnScanCacheKey, bool>>>,
    object_store: Arc<dyn ObjectStore>,
    catalog_root: PathBuf,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct I128ColumnMinMaxCacheKey {
    path: PathBuf,
    len: u64,
    modified_nanos: Option<u128>,
    column: String,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct MonotonicColumnScanCacheKey {
    path: PathBuf,
    len: u64,
    modified_nanos: Option<u128>,
    column: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LocalExecutionOptions {
    pub shuffle_file_target_bytes: u64,
}

impl Default for LocalExecutionOptions {
    fn default() -> Self {
        Self {
            shuffle_file_target_bytes: LOCAL_SHUFFLE_FILE_TARGET_BYTES,
        }
    }
}

pub struct LocalExecutionGraphOutput {
    pub streams: Vec<SendableBatchStream>,
    pub metrics: LocalExecutionGraphMetrics,
    pub stage_metrics: Vec<LocalStageExecutionMetrics>,
}

#[derive(Debug, Default, Clone)]
pub(crate) struct RowGroupBatchScanProfile {
    pub projected_columns: usize,
    pub row_groups_total: usize,
    pub row_groups_scanned: usize,
    pub compressed_bytes_total: u64,
    pub compressed_bytes_scanned: u64,
    pub metadata_nanos: u64,
    pub planning_nanos: u64,
    pub next_nanos: u64,
    pub max_next_nanos: u64,
    pub p95_next_nanos: u64,
    pub next_calls: usize,
    pub eof_calls: usize,
    pub output_batches: usize,
    pub output_rows: usize,
    pub zero_row_batches: usize,
}

impl RowGroupBatchScanProfile {
    pub(crate) fn merge_from(&mut self, other: Self) {
        self.projected_columns = self.projected_columns.max(other.projected_columns);
        self.row_groups_total = self.row_groups_total.max(other.row_groups_total);
        self.row_groups_scanned = self
            .row_groups_scanned
            .saturating_add(other.row_groups_scanned);
        self.compressed_bytes_total = self
            .compressed_bytes_total
            .max(other.compressed_bytes_total);
        self.compressed_bytes_scanned = self
            .compressed_bytes_scanned
            .saturating_add(other.compressed_bytes_scanned);
        self.metadata_nanos = self.metadata_nanos.saturating_add(other.metadata_nanos);
        self.planning_nanos = self.planning_nanos.saturating_add(other.planning_nanos);
        self.next_nanos = self.next_nanos.saturating_add(other.next_nanos);
        self.max_next_nanos = self.max_next_nanos.max(other.max_next_nanos);
        self.p95_next_nanos = self.p95_next_nanos.max(other.p95_next_nanos);
        self.next_calls = self.next_calls.saturating_add(other.next_calls);
        self.eof_calls = self.eof_calls.saturating_add(other.eof_calls);
        self.output_batches = self.output_batches.saturating_add(other.output_batches);
        self.output_rows = self.output_rows.saturating_add(other.output_rows);
        self.zero_row_batches = self.zero_row_batches.saturating_add(other.zero_row_batches);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DirectPrimitiveKeyType {
    I32,
    I64,
    DictionaryI32Utf8,
}

impl DirectPrimitiveKeyType {
    fn column_type_descriptor(self) -> &'static str {
        match self {
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::DictionaryI32Utf8 => "dictionary_i32_utf8",
        }
    }
}

fn partition_row_groups_balanced(row_groups: &[usize], partitions: usize) -> Vec<Vec<usize>> {
    if !std::env::var("DODAM_ENABLE_BALANCED_DIRECT_PRIMITIVE_PARTITIONS")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
    {
        let partitions = partitions.min(row_groups.len()).max(1);
        let chunk_size = row_groups.len().div_ceil(partitions).max(1);
        return row_groups
            .chunks(chunk_size)
            .map(|chunk| chunk.to_vec())
            .collect();
    }
    let partitions = partitions.min(row_groups.len()).max(1);
    let base = row_groups.len() / partitions;
    let remainder = row_groups.len() % partitions;
    let mut output = Vec::with_capacity(partitions);
    let mut cursor = 0usize;
    for partition in 0..partitions {
        let len = base + usize::from(partition < remainder);
        output.push(row_groups[cursor..cursor + len].to_vec());
        cursor += len;
    }
    output
}

fn fused_dictionary_selected_workers(row_groups: usize) -> usize {
    let default_workers = choose_parallel_workers(WorkerCostInput {
        row_groups,
        available_parallelism: std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(4),
        max_workers: usize::MAX,
    });
    std::env::var("DODAM_FUSED_DICT_SELECTED_WORKERS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default_workers)
        .min(row_groups.max(1))
}

#[derive(Debug, Clone)]
struct OwnedDirectPrimitiveColumnSpec {
    name: String,
    column_type: DirectPrimitiveColumnType,
}

impl OwnedDirectPrimitiveColumnSpec {
    fn borrowed_specs(columns: &[Self]) -> Vec<DirectPrimitiveColumnSpec<'_>> {
        columns
            .iter()
            .map(|column| DirectPrimitiveColumnSpec {
                name: column.name.as_str(),
                column_type: column.column_type,
            })
            .collect()
    }
}

#[derive(Debug, Clone)]
struct DirectUtf8CountSumShape {
    predicate_column: String,
    group_column: String,
    sum_column: String,
    op: ComparisonOp,
    value: i32,
}

#[derive(Default)]
struct DirectUtf8CountSumState {
    groups: HashMap<Vec<u8>, usize>,
    group_values: Vec<(Vec<u8>, u64, i64)>,
    null_group: Option<(u64, i64)>,
    local_dictionary_groups: Vec<(u64, i64)>,
    local_dictionary_touched: Vec<usize>,
    batches: usize,
    rows: usize,
}

impl DirectUtf8CountSumState {
    fn group_id(&mut self, key: &[u8]) -> usize {
        if let Some(group_id) = self.groups.get(key).copied() {
            return group_id;
        }
        let group_id = self.group_values.len();
        let key = key.to_vec();
        self.groups.insert(key.clone(), group_id);
        self.group_values.push((key, 0, 0));
        group_id
    }

    fn update_group(&mut self, key: &[u8], count: u64, sum: i64) {
        let group_id = self.group_id(key);
        let (_, group_count, group_sum) = &mut self.group_values[group_id];
        *group_count = group_count.saturating_add(count);
        *group_sum = group_sum.saturating_add(sum);
    }

    fn consume(
        &mut self,
        predicate_values: &[i32],
        sum_values: &[i64],
        group_def_levels: &[i16],
        group_values: &[parquet::data_type::ByteArray],
        shape: &DirectUtf8CountSumShape,
    ) {
        self.batches += 1;
        let mut group_index = 0usize;
        for row in 0..predicate_values.len() {
            let group_is_null = group_def_levels[row] == 0;
            if !direct_i32_predicate_matches(predicate_values[row], shape.op, shape.value) {
                if !group_is_null {
                    group_index += 1;
                }
                continue;
            }
            self.rows += 1;
            if group_is_null {
                let entry = self.null_group.get_or_insert((0, 0));
                entry.0 = entry.0.saturating_add(1);
                entry.1 = entry.1.saturating_add(sum_values[row]);
            } else {
                let group = group_values[group_index].data();
                group_index += 1;
                self.update_group(group, 1, sum_values[row]);
            }
        }
    }

    fn consume_dictionary_ids(
        &mut self,
        predicate_values: &[i32],
        sum_values: &[i64],
        group_def_levels: Option<&[i16]>,
        group_ids: &[i32],
        dictionary: &[bytes::Bytes],
        shape: &DirectUtf8CountSumShape,
    ) {
        self.batches += 1;
        let mut group_index = 0usize;
        if self.local_dictionary_groups.len() < dictionary.len() {
            self.local_dictionary_groups
                .resize(dictionary.len(), (0, 0));
        }
        self.local_dictionary_touched.clear();
        for row in 0..predicate_values.len() {
            let group_is_null = group_def_levels.is_some_and(|levels| levels[row] == 0);
            if !direct_i32_predicate_matches(predicate_values[row], shape.op, shape.value) {
                if !group_is_null {
                    group_index += 1;
                }
                continue;
            }
            self.rows += 1;
            if group_is_null {
                let entry = self.null_group.get_or_insert((0, 0));
                entry.0 = entry.0.saturating_add(1);
                entry.1 = entry.1.saturating_add(sum_values[row]);
            } else {
                let id = group_ids[group_index] as usize;
                group_index += 1;
                let entry = &mut self.local_dictionary_groups[id];
                if entry.0 == 0 {
                    self.local_dictionary_touched.push(id);
                }
                entry.0 = entry.0.saturating_add(1);
                entry.1 = entry.1.saturating_add(sum_values[row]);
            }
        }
        let touched = self.local_dictionary_touched.drain(..).collect::<Vec<_>>();
        for id in touched {
            let entry = &mut self.local_dictionary_groups[id];
            let (count, sum) = *entry;
            *entry = (0, 0);
            self.update_group(dictionary[id].as_ref(), count, sum);
        }
    }

    fn finish(
        self,
        fragments: usize,
        count_expr: AggregateExpr,
        sum_expr: AggregateExpr,
    ) -> Result<AggregateMetrics> {
        let mut groups =
            Vec::with_capacity(self.group_values.len() + usize::from(self.null_group.is_some()));
        if let Some((count, sum)) = self.null_group {
            groups.push(GroupAggregateResult {
                keys: vec![crate::execution::GroupValue::Utf8(None)],
                values: vec![
                    crate::execution::AggregateResult {
                        expr: count_expr.clone(),
                        value: crate::execution::AggregateValue::Count(count),
                    },
                    crate::execution::AggregateResult {
                        expr: sum_expr.clone(),
                        value: crate::execution::AggregateValue::Int64(Some(sum)),
                    },
                ],
            });
        }
        for (key, count, sum) in self.group_values {
            let key = String::from_utf8(key).map_err(|_| {
                DodamError::UnsupportedSql("invalid UTF8 group key in direct aggregate".to_string())
            })?;
            groups.push(GroupAggregateResult {
                keys: vec![crate::execution::GroupValue::Utf8(Some(key))],
                values: vec![
                    crate::execution::AggregateResult {
                        expr: count_expr.clone(),
                        value: crate::execution::AggregateValue::Count(count),
                    },
                    crate::execution::AggregateResult {
                        expr: sum_expr.clone(),
                        value: crate::execution::AggregateValue::Int64(Some(sum)),
                    },
                ],
            });
        }
        groups.sort_by(|left, right| direct_utf8_group_key_cmp(&left.keys[0], &right.keys[0]));
        Ok(AggregateMetrics {
            fragments,
            batches: self.batches,
            rows: self.rows,
            groups,
            ..AggregateMetrics::default()
        })
    }
}

fn direct_utf8_group_key_cmp(
    left: &crate::execution::GroupValue,
    right: &crate::execution::GroupValue,
) -> std::cmp::Ordering {
    match (left, right) {
        (crate::execution::GroupValue::Utf8(None), crate::execution::GroupValue::Utf8(None)) => {
            std::cmp::Ordering::Equal
        }
        (crate::execution::GroupValue::Utf8(None), _) => std::cmp::Ordering::Less,
        (_, crate::execution::GroupValue::Utf8(None)) => std::cmp::Ordering::Greater,
        (
            crate::execution::GroupValue::Utf8(Some(left)),
            crate::execution::GroupValue::Utf8(Some(right)),
        ) => left.cmp(right),
        _ => std::cmp::Ordering::Equal,
    }
}

fn direct_i32_predicate_matches(value: i32, op: ComparisonOp, bound: i32) -> bool {
    match op {
        ComparisonOp::Eq => value == bound,
        ComparisonOp::NotEq => value != bound,
        ComparisonOp::Lt => value < bound,
        ComparisonOp::LtEq => value <= bound,
        ComparisonOp::Gt => value > bound,
        ComparisonOp::GtEq => value >= bound,
    }
}

fn direct_i32_utf8_count_sum_shape(
    aggregates: &[AggregateExpr],
    group_by: &[String],
    filter: &FilterExpr,
) -> Result<Option<DirectUtf8CountSumShape>> {
    if group_by.len() != 1 {
        return Ok(None);
    }
    let [
        AggregateExpr::CountStar | AggregateExpr::Count(_),
        AggregateExpr::Sum(sum_column),
    ] = aggregates
    else {
        return Ok(None);
    };
    let Expr::Comparison(ComparisonExpr { column, op, value }) = filter.expr() else {
        return Ok(None);
    };
    let LiteralValue::Int64(value) = value else {
        return Ok(None);
    };
    let Ok(value) = i32::try_from(*value) else {
        return Ok(None);
    };
    Ok(Some(DirectUtf8CountSumShape {
        predicate_column: column.clone(),
        group_column: group_by[0].clone(),
        sum_column: sum_column.clone(),
        op: *op,
        value,
    }))
}

#[derive(Debug, Clone)]
struct DirectPrimitiveFoldPlan {
    path: PathBuf,
    batch_size: usize,
    row_groups: Vec<usize>,
    columns: Vec<OwnedDirectPrimitiveColumnSpec>,
}

impl DirectPrimitiveFoldPlan {
    fn new(
        path: impl Into<PathBuf>,
        batch_size: usize,
        row_groups: Vec<usize>,
        columns: Vec<OwnedDirectPrimitiveColumnSpec>,
    ) -> Self {
        Self {
            path: path.into(),
            batch_size,
            row_groups,
            columns,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DirectPrimitiveBatchLocation {
    pub row_group: usize,
    pub row_offset: usize,
}

#[derive(Debug, Clone)]
pub struct OrderedRowGroupBoundary<K, State> {
    pub key: K,
    pub state: State,
}

#[derive(Debug, Clone)]
pub struct OrderedRowGroupChunk<K, State, Output> {
    pub output: Output,
    pub first: Option<OrderedRowGroupBoundary<K, State>>,
    pub last: Option<OrderedRowGroupBoundary<K, State>>,
}

pub fn merge_ordered_row_group_chunks<K, State, Output, MergeOutput, MergeState, EmitState>(
    chunks: Vec<OrderedRowGroupChunk<K, State, Output>>,
    output: &mut Output,
    mut merge_output: MergeOutput,
    mut merge_state: MergeState,
    mut emit_state: EmitState,
) where
    K: Eq,
    MergeOutput: FnMut(&mut Output, Output),
    MergeState: FnMut(&mut State, State),
    EmitState: FnMut(&mut Output, OrderedRowGroupBoundary<K, State>),
{
    let mut pending = None::<OrderedRowGroupBoundary<K, State>>;
    for chunk in chunks {
        if let Some(first) = chunk.first {
            merge_ordered_row_group_boundary(
                output,
                &mut pending,
                first,
                chunk.last.is_some(),
                &mut merge_state,
                &mut emit_state,
            );
        }
        merge_output(output, chunk.output);
        if let Some(last) = chunk.last {
            pending = Some(last);
        }
    }
    if let Some(boundary) = pending {
        emit_state(output, boundary);
    }
}

fn merge_ordered_row_group_boundary<K, State, Output, MergeState, EmitState>(
    output: &mut Output,
    pending: &mut Option<OrderedRowGroupBoundary<K, State>>,
    boundary: OrderedRowGroupBoundary<K, State>,
    complete_in_chunk: bool,
    merge_state: &mut MergeState,
    emit_state: &mut EmitState,
) where
    K: Eq,
    MergeState: FnMut(&mut State, State),
    EmitState: FnMut(&mut Output, OrderedRowGroupBoundary<K, State>),
{
    if let Some(mut existing) = pending.take() {
        if existing.key == boundary.key {
            merge_state(&mut existing.state, boundary.state);
            if complete_in_chunk {
                emit_state(output, existing);
            } else {
                *pending = Some(existing);
            }
            return;
        }
        emit_state(output, existing);
    }
    if complete_in_chunk {
        emit_state(output, boundary);
    } else {
        *pending = Some(boundary);
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LocalExecutionGraphMetrics {
    pub stages_executed: usize,
    pub tasks_executed: usize,
    pub task_output_batches: usize,
    pub task_output_rows: usize,
    pub shuffle_write_files: usize,
    pub shuffle_write_batches: usize,
    pub shuffle_write_rows: usize,
    pub shuffle_write_bytes: u64,
    pub shuffle_read_files: usize,
    pub shuffle_read_batches: usize,
    pub shuffle_read_rows: usize,
    pub shuffle_read_bytes: u64,
    pub task_execution_nanos: u64,
    pub shuffle_repartition_nanos: u64,
    pub shuffle_write_nanos: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LocalStageExecutionMetrics {
    pub stage_id: usize,
    pub tasks_executed: usize,
    pub task_output_batches: usize,
    pub task_output_rows: usize,
    pub shuffle_write_files: usize,
    pub shuffle_write_batches: usize,
    pub shuffle_write_rows: usize,
    pub shuffle_write_bytes: u64,
    pub shuffle_read_files: usize,
    pub shuffle_read_batches: usize,
    pub shuffle_read_rows: usize,
    pub shuffle_read_bytes: u64,
    pub task_execution_nanos: u64,
    pub shuffle_repartition_nanos: u64,
    pub shuffle_write_nanos: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct LocalShuffleWriteMetrics {
    files: usize,
    batches: usize,
    rows: usize,
    bytes: u64,
    write_nanos: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct LocalShuffleReadMetrics {
    files: usize,
    batches: usize,
    rows: usize,
    bytes: u64,
}

#[derive(Debug, Clone, Copy, Default)]
struct ParquetMapChunkMetrics {
    chunks: usize,
    row_groups: usize,
    projected_columns: usize,
    rows: usize,
    batches: usize,
    zero_batches: usize,
    total_nanos: u64,
    read_nanos: u64,
    reader_next_nanos: u64,
    reader_p95_next_nanos: u64,
    reader_max_next_nanos: u64,
    reader_calls: usize,
    reader_eof: usize,
    consume_nanos: u64,
    compressed_bytes_scanned: u64,
    compressed_bytes_total: u64,
}

impl ParquetMapChunkMetrics {
    fn add(&mut self, other: Self) {
        self.chunks = self.chunks.saturating_add(other.chunks);
        self.row_groups = self.row_groups.saturating_add(other.row_groups);
        self.projected_columns = self.projected_columns.max(other.projected_columns);
        self.rows = self.rows.saturating_add(other.rows);
        self.batches = self.batches.saturating_add(other.batches);
        self.zero_batches = self.zero_batches.saturating_add(other.zero_batches);
        self.total_nanos = self.total_nanos.saturating_add(other.total_nanos);
        self.read_nanos = self.read_nanos.saturating_add(other.read_nanos);
        self.reader_next_nanos = self
            .reader_next_nanos
            .saturating_add(other.reader_next_nanos);
        self.reader_p95_next_nanos = self.reader_p95_next_nanos.max(other.reader_p95_next_nanos);
        self.reader_max_next_nanos = self.reader_max_next_nanos.max(other.reader_max_next_nanos);
        self.reader_calls = self.reader_calls.saturating_add(other.reader_calls);
        self.reader_eof = self.reader_eof.saturating_add(other.reader_eof);
        self.consume_nanos = self.consume_nanos.saturating_add(other.consume_nanos);
        self.compressed_bytes_scanned = self
            .compressed_bytes_scanned
            .saturating_add(other.compressed_bytes_scanned);
        self.compressed_bytes_total = self
            .compressed_bytes_total
            .max(other.compressed_bytes_total);
    }
}

struct ParquetMapChunkResult<Output> {
    output: Output,
    metrics: ParquetMapChunkMetrics,
}

struct ParquetMapResults<Output> {
    outputs: Vec<Output>,
    metrics: ParquetMapChunkMetrics,
}

impl LocalExecutionGraphMetrics {
    fn add_shuffle_write(&mut self, metrics: LocalShuffleWriteMetrics) {
        self.shuffle_write_files = self.shuffle_write_files.saturating_add(metrics.files);
        self.shuffle_write_batches = self.shuffle_write_batches.saturating_add(metrics.batches);
        self.shuffle_write_rows = self.shuffle_write_rows.saturating_add(metrics.rows);
        self.shuffle_write_bytes = self.shuffle_write_bytes.saturating_add(metrics.bytes);
        self.shuffle_write_nanos = self.shuffle_write_nanos.saturating_add(metrics.write_nanos);
    }
}

impl LocalStageExecutionMetrics {
    fn add_task_output(&mut self, batches: &[RecordBatch], elapsed: Duration) {
        self.tasks_executed = self.tasks_executed.saturating_add(1);
        self.task_output_batches = self.task_output_batches.saturating_add(batches.len());
        self.task_output_rows = self
            .task_output_rows
            .saturating_add(batches.iter().map(RecordBatch::num_rows).sum::<usize>());
        self.task_execution_nanos = self
            .task_execution_nanos
            .saturating_add(elapsed_nanos(elapsed));
    }

    fn add_shuffle_read(&mut self, metrics: LocalShuffleReadMetrics) {
        self.shuffle_read_files = self.shuffle_read_files.saturating_add(metrics.files);
        self.shuffle_read_batches = self.shuffle_read_batches.saturating_add(metrics.batches);
        self.shuffle_read_rows = self.shuffle_read_rows.saturating_add(metrics.rows);
        self.shuffle_read_bytes = self.shuffle_read_bytes.saturating_add(metrics.bytes);
    }

    fn add_shuffle_write(&mut self, metrics: LocalShuffleWriteMetrics) {
        self.shuffle_write_files = self.shuffle_write_files.saturating_add(metrics.files);
        self.shuffle_write_batches = self.shuffle_write_batches.saturating_add(metrics.batches);
        self.shuffle_write_rows = self.shuffle_write_rows.saturating_add(metrics.rows);
        self.shuffle_write_bytes = self.shuffle_write_bytes.saturating_add(metrics.bytes);
        self.shuffle_write_nanos = self.shuffle_write_nanos.saturating_add(metrics.write_nanos);
    }
}

#[derive(Debug, Clone)]
pub struct JoinParquetRequest {
    pub left_path: PathBuf,
    pub right_path: PathBuf,
    pub batch_size: usize,
    pub left_keys: Vec<String>,
    pub right_keys: Vec<String>,
    pub left_prefix: String,
    pub right_prefix: String,
    pub left_projection: Projection,
    pub right_projection: Projection,
    pub left_filter: Option<FilterExpr>,
    pub right_filter: Option<FilterExpr>,
    pub output_projection: Projection,
    pub join_memory_limit_bytes: u64,
    pub join_algorithm: JoinAlgorithm,
    pub join_type: JoinType,
}

#[derive(Debug, Clone)]
pub struct JoinTableRequest {
    pub left: TableScanSource,
    pub right: TableScanSource,
    pub batch_size: usize,
    pub left_keys: Vec<String>,
    pub right_keys: Vec<String>,
    pub left_prefix: String,
    pub right_prefix: String,
    pub left_projection: Projection,
    pub right_projection: Projection,
    pub left_filter: Option<FilterExpr>,
    pub right_filter: Option<FilterExpr>,
    pub output_projection: Projection,
    pub join_memory_limit_bytes: u64,
    pub join_algorithm: JoinAlgorithm,
    pub join_type: JoinType,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum JoinAlgorithm {
    #[default]
    Auto,
    SortMerge,
}

#[derive(Debug, Clone)]
pub struct ScanPlan {
    pub source: TableScanSource,
    pub batch_size: usize,
    pub limit: Option<usize>,
    pub output_projection: Projection,
    pub scan_projection: Projection,
    pub filter: Option<FilterExpr>,
    pub residual_filter: Option<FilterExpr>,
    pub pushdown_predicates: Vec<Expr>,
    pub row_filter_predicates: Vec<Expr>,
    pub has_filter: bool,
    pub distinct: bool,
    pub order_by: Option<SortKey>,
    pub estimated_bytes: u64,
    pub operators: Vec<ScanOperator>,
    pub preserve_order: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanOperator {
    Limit,
    Sort,
    Distinct,
    Projection,
    Filter,
    Scan,
}

#[derive(Debug, Clone)]
struct ScanPlanOptions {
    batch_size: usize,
    limit: Option<usize>,
    projection: Projection,
    filter: Option<FilterExpr>,
    order_by: Option<SortKey>,
    distinct: bool,
}

impl ScanPlan {
    pub fn explain(&self) -> String {
        self.to_plan_node().render_text()
    }

    pub fn to_logical_plan(&self) -> LogicalPlan {
        LogicalPlan::TableScan(LogicalScan {
            source: plan_table_source_from_scan_source(&self.source),
            batch_size: self.batch_size,
            projection: self.output_projection.clone(),
            filter: self.filter.clone(),
            order_by: self.order_by.clone(),
            limit: self.limit,
            distinct: self.distinct,
        })
    }

    pub fn to_plan_node(&self) -> PhysicalPlanNode {
        let mut current = None;
        for operator in self.operators.iter().rev() {
            let mut node = match operator {
                ScanOperator::Scan => PhysicalPlanNode::new("ScanExec")
                    .attr("format", format!("{:?}", self.source.format))
                    .attr("fragments", self.source.fragments.len())
                    .attr("rows", self.source.statistics.rows)
                    .attr("row_groups", self.source.statistics.row_groups)
                    .attr("compressed_bytes", self.source.statistics.compressed_bytes)
                    .attr("estimated_bytes", self.estimated_bytes)
                    .attr("batch_size", self.batch_size)
                    .attr("scan_projection", projection_display(&self.scan_projection))
                    .attr("pushdown_predicates", self.pushdown_predicates.len()),
                ScanOperator::Filter if self.residual_filter.is_some() => {
                    PhysicalPlanNode::new("FilterExec").attr("predicate", "residual")
                }
                ScanOperator::Filter if self.has_filter => {
                    PhysicalPlanNode::new("FilterExec").attr("predicate", "pushdown_only")
                }
                ScanOperator::Filter => continue,
                ScanOperator::Projection => PhysicalPlanNode::new("ProjectionExec")
                    .attr("projection", projection_display(&self.output_projection)),
                ScanOperator::Distinct => PhysicalPlanNode::new("DistinctExec"),
                ScanOperator::Sort => {
                    let Some(order_by) = &self.order_by else {
                        continue;
                    };
                    PhysicalPlanNode::new("SortExec")
                        .attr("order_by", format!("[{}]", sort_key_display(order_by)))
                        .attr(
                            "limit",
                            self.limit
                                .map(|limit| limit.to_string())
                                .unwrap_or_else(|| "none".to_string()),
                        )
                }
                ScanOperator::Limit => {
                    let Some(limit) = self.limit else {
                        continue;
                    };
                    PhysicalPlanNode::new("LimitExec").attr("limit", limit)
                }
            };
            if let Some(child) = current {
                node = node.child(child);
            }
            current = Some(node);
        }
        current.unwrap_or_else(|| PhysicalPlanNode::new("EmptyExec"))
    }
}

#[derive(Debug, Clone)]
pub struct AggregatePlan {
    pub scan: ScanPlan,
    pub aggregates: Vec<AggregateExpr>,
    pub group_by: Vec<String>,
    pub column_read_plan: AggregateColumnReadPlan,
    pub direct_physical: Option<PhysicalPlanNode>,
}

impl AggregatePlan {
    pub fn explain(&self) -> String {
        self.to_plan_node().render_text()
    }

    pub fn to_logical_plan(&self) -> LogicalPlan {
        LogicalPlan::Aggregate {
            input: Box::new(self.scan.to_logical_plan()),
            aggregates: self.aggregates.clone(),
            group_by: self.group_by.clone(),
        }
    }

    pub fn to_plan_node(&self) -> PhysicalPlanNode {
        if let Some(plan) = &self.direct_physical {
            return plan.clone();
        }
        let local_fold = PhysicalPlanNode::new("LocalFoldExec")
            .attr(
                "mode",
                if self.group_by.is_empty() {
                    "global"
                } else {
                    "grouped"
                },
            )
            .attr("group_by", format!("[{}]", self.group_by.join(",")))
            .attr(
                "payload_columns",
                projection_display(&self.column_read_plan.payload_projection),
            )
            .attr(
                "predicate_columns",
                projection_display(&self.column_read_plan.predicate_projection),
            )
            .attr(
                "scan_columns",
                projection_display(&self.column_read_plan.scan_projection),
            )
            .attr(
                "aggregates",
                format!(
                    "[{}]",
                    self.aggregates
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join(",")
                ),
            )
            .execution(PhysicalExecutionConfig::LocalFold {
                group_by: self.group_by.clone(),
                aggregates: self.aggregates.clone(),
            })
            .child(self.scan.to_plan_node());
        PhysicalPlanNode::new("FinalMergeExec")
            .attr(
                "mode",
                if self.group_by.is_empty() {
                    "global"
                } else {
                    "grouped"
                },
            )
            .attr("group_by", format!("[{}]", self.group_by.join(",")))
            .attr(
                "payload_columns",
                projection_display(&self.column_read_plan.payload_projection),
            )
            .attr(
                "predicate_columns",
                projection_display(&self.column_read_plan.predicate_projection),
            )
            .attr(
                "scan_columns",
                projection_display(&self.column_read_plan.scan_projection),
            )
            .attr(
                "aggregates",
                format!(
                    "[{}]",
                    self.aggregates
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join(",")
                ),
            )
            .execution(PhysicalExecutionConfig::FinalMerge {
                group_by: self.group_by.clone(),
                aggregates: self.aggregates.clone(),
            })
            .child(local_fold)
    }
}

#[derive(Debug, Clone)]
pub struct AggregateColumnReadPlan {
    pub payload_projection: Projection,
    pub predicate_projection: Projection,
    pub scan_projection: Projection,
}

#[derive(Debug, Clone)]
pub struct JoinPlan {
    pub request: JoinTablePlanRequest,
    pub left_scan: ScanPlan,
    pub right_scan: ScanPlan,
    pub strategy: JoinExecutionStrategy,
}

#[derive(Debug, Clone)]
pub struct JoinTablePlanRequest {
    pub batch_size: usize,
    pub left_keys: Vec<String>,
    pub right_keys: Vec<String>,
    pub left_prefix: String,
    pub right_prefix: String,
    pub left_projection: Projection,
    pub right_projection: Projection,
    pub left_filter: Option<FilterExpr>,
    pub right_filter: Option<FilterExpr>,
    pub output_projection: Projection,
    pub join_memory_limit_bytes: u64,
    pub join_algorithm: JoinAlgorithm,
    pub join_type: JoinType,
}

impl JoinPlan {
    pub fn explain(&self) -> String {
        self.to_plan_node().render_text()
    }

    pub fn to_logical_plan(&self) -> LogicalPlan {
        LogicalPlan::Join {
            left: Box::new(self.left_scan.to_logical_plan()),
            right: Box::new(self.right_scan.to_logical_plan()),
            join_type: self.request.join_type,
            left_keys: self.request.left_keys.clone(),
            right_keys: self.request.right_keys.clone(),
            left_prefix: self.request.left_prefix.clone(),
            right_prefix: self.request.right_prefix.clone(),
            output_projection: self.request.output_projection.clone(),
        }
    }

    pub fn physical_strategy(&self) -> PhysicalJoinStrategy {
        match self.strategy {
            JoinExecutionStrategy::Hash { build_side } => PhysicalJoinStrategy::Hash { build_side },
            JoinExecutionStrategy::PartitionedHash {
                partitions,
                memory_limit_bytes,
            } => PhysicalJoinStrategy::PartitionedHash {
                partitions,
                memory_limit_bytes,
            },
            JoinExecutionStrategy::SortMerge => PhysicalJoinStrategy::SortMerge,
        }
    }

    pub fn to_plan_node(&self) -> PhysicalPlanNode {
        let mut node = PhysicalPlanNode::new("JoinExec")
            .attr("type", format!("{:?}", self.request.join_type))
            .attr("strategy", self.strategy.name())
            .attr(
                "left_keys",
                format!("[{}]", self.request.left_keys.join(",")),
            )
            .attr(
                "right_keys",
                format!("[{}]", self.request.right_keys.join(",")),
            )
            .attr("estimated_left_bytes", self.left_scan.estimated_bytes)
            .attr("estimated_right_bytes", self.right_scan.estimated_bytes);
        if let JoinExecutionStrategy::Hash { build_side } = self.strategy {
            node = node.attr("build", format!("{build_side:?}"));
        }
        if let JoinExecutionStrategy::PartitionedHash {
            partitions,
            memory_limit_bytes,
        } = self.strategy
        {
            node = node
                .attr("partitions", partitions)
                .attr("memory_limit_bytes", memory_limit_bytes);
        }
        node.child(self.left_scan.to_plan_node().attr("side", "left"))
            .child(self.right_scan.to_plan_node().attr("side", "right"))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinExecutionStrategy {
    Hash {
        build_side: JoinBuildSide,
    },
    PartitionedHash {
        partitions: usize,
        memory_limit_bytes: u64,
    },
    SortMerge,
}

impl JoinExecutionStrategy {
    pub fn name(self) -> &'static str {
        match self {
            Self::Hash { .. } => "hash",
            Self::PartitionedHash { .. } => "partitioned_hash",
            Self::SortMerge => "sort_merge",
        }
    }
}

impl Default for DodamEngine {
    fn default() -> Self {
        Self {
            metadata_cache: Arc::new(ParquetMetadataCache::default()),
            file_cache: Arc::new(ParquetFileCache::default()),
            i128_column_min_max_cache: Arc::new(Mutex::new(HashMap::new())),
            monotonic_column_scan_cache: Arc::new(Mutex::new(HashMap::new())),
            object_store: Arc::new(LocalFileSystemObjectStore),
            catalog_root: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
        }
    }
}

impl DodamEngine {
    pub fn with_catalog_root(mut self, catalog_root: impl Into<PathBuf>) -> Self {
        self.catalog_root = catalog_root.into();
        self
    }

    pub fn metadata_cache_len(&self) -> usize {
        self.metadata_cache.len()
    }

    pub fn file_cache_len(&self) -> usize {
        self.file_cache.len()
    }

    pub fn file_cache_bytes(&self) -> usize {
        self.file_cache.bytes()
    }

    pub fn file_cache_stats(&self) -> ParquetFileCacheStats {
        self.file_cache.stats()
    }

    pub fn parquet_row_group_count(&self, path: impl AsRef<Path>) -> Result<usize> {
        parquet_row_group_count_with_store(
            path.as_ref(),
            self.file_cache.clone(),
            self.object_store.as_ref(),
        )
    }

    pub fn parquet_pruned_row_groups(
        &self,
        path: impl AsRef<Path>,
        projection: &Projection,
        pruning_predicates: &[Expr],
    ) -> Result<Vec<usize>> {
        Ok(plan_parquet_scan_tasks(
            path,
            projection,
            pruning_predicates,
            &self.metadata_cache,
            self.object_store.as_ref(),
        )?
        .tasks
        .into_iter()
        .map(|task| task.row_group)
        .collect())
    }

    pub fn parquet_total_row_count(&self, path: impl AsRef<Path>) -> Result<usize> {
        parquet_total_row_count_with_store(
            path.as_ref(),
            self.file_cache.clone(),
            self.object_store.as_ref(),
        )
    }

    pub(crate) fn parquet_primitive_column_min_max_by_row_group(
        &self,
        path: impl AsRef<Path>,
        column: &str,
    ) -> Result<Option<Vec<PrimitiveRowGroupMinMax>>> {
        read_parquet_primitive_column_min_max_by_row_group(
            path,
            column,
            &self.metadata_cache,
            self.object_store.as_ref(),
        )
    }

    pub(crate) fn parquet_direct_primitive_column_types(
        &self,
        path: impl AsRef<Path>,
        columns: &[String],
    ) -> Result<Option<Vec<DirectPrimitiveColumnType>>> {
        let file = self.object_store.open(path.as_ref())?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let parquet_columns = reader
            .metadata()
            .file_metadata()
            .schema_descr()
            .columns()
            .to_vec();
        columns
            .iter()
            .map(|name| {
                let Some(column) = parquet_columns.iter().find(|column| column.name() == name)
                else {
                    return Ok(None);
                };
                let column_type = match column.physical_type() {
                    parquet::basic::Type::INT32 => DirectPrimitiveColumnType::I32,
                    parquet::basic::Type::INT64 => DirectPrimitiveColumnType::I64,
                    _ => return Ok(None),
                };
                Ok(Some(column_type))
            })
            .collect()
    }

    pub(crate) fn scan_parquet_primitive_columns_view<F>(
        &self,
        path: impl AsRef<Path>,
        batch_size: usize,
        row_groups: &[usize],
        columns: &[DirectPrimitiveColumnSpec<'_>],
        mut consume: F,
    ) -> Result<Option<DirectPrimitiveColumnScanMetrics>>
    where
        F: for<'a> FnMut(BatchView<'a>) -> Result<()>,
    {
        scan_parquet_primitive_columns_with_store(
            path.as_ref(),
            batch_size,
            row_groups,
            columns,
            self.file_cache.clone(),
            self.object_store.as_ref(),
            move |columns| consume(BatchView::from_raw_columns(columns)),
        )
    }

    pub(crate) fn scan_parquet_primitive_columns_page_view<F>(
        &self,
        path: impl AsRef<Path>,
        batch_size: usize,
        row_groups: &[usize],
        columns: &[DirectPrimitiveColumnSpec<'_>],
        mut consume: F,
    ) -> Result<Option<DirectPrimitiveColumnScanMetrics>>
    where
        F: for<'a> FnMut(BatchView<'a>) -> Result<()>,
    {
        scan_parquet_primitive_columns_with_store_page_reader(
            path.as_ref(),
            batch_size,
            row_groups,
            columns,
            self.file_cache.clone(),
            self.object_store.as_ref(),
            move |columns| consume(BatchView::from_raw_columns(columns)),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn scan_parquet_dictionary_date_range_selected_primitive_columns_parallel<F>(
        &self,
        path: impl Into<PathBuf>,
        row_groups: Vec<usize>,
        predicate_column: String,
        start_days: i32,
        end_days: i32,
        payload_columns: Vec<(String, DirectPrimitiveColumnType)>,
        consume: F,
    ) -> Result<Option<DirectPrimitiveColumnScanMetrics>>
    where
        F: for<'a> Fn(BatchView<'a>) -> Result<()> + Sync,
    {
        let path = path.into();
        let payload_columns = payload_columns
            .into_iter()
            .map(|(name, column_type)| OwnedDirectPrimitiveColumnSpec { name, column_type })
            .collect::<Vec<_>>();
        let mut scan_metrics = DirectPrimitiveColumnScanMetrics::default();
        if row_groups.is_empty() {
            return Ok(Some(scan_metrics));
        }
        let workers = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(4)
            .min(row_groups.len());
        let (sender, receiver) = mpsc::channel();
        let row_group_partitions = partition_row_groups_balanced(&row_groups, workers);
        std::thread::scope(|scope| {
            for row_group_partition in row_group_partitions {
                let sender = sender.clone();
                let engine = self.clone();
                let path = path.clone();
                let predicate_column = predicate_column.clone();
                let payload_columns = payload_columns.clone();
                let consume = &consume;
                scope.spawn(move || {
                    let specs = OwnedDirectPrimitiveColumnSpec::borrowed_specs(&payload_columns);
                    let result =
                        scan_parquet_dictionary_date_range_selected_primitive_columns_with_store(
                            &path,
                            &row_group_partition,
                            &predicate_column,
                            start_days,
                            end_days,
                            &specs,
                            engine.file_cache.clone(),
                            engine.object_store.as_ref(),
                            |columns| consume(BatchView::from_raw_columns(columns)),
                        );
                    let _ = sender.send(result);
                });
            }
        });
        drop(sender);
        for received in receiver {
            let Some(metrics) = received? else {
                return Ok(None);
            };
            scan_metrics.merge_from(metrics);
        }
        let mut profile_columns = Vec::with_capacity(payload_columns.len() + 1);
        profile_columns.push(OwnedDirectPrimitiveColumnSpec {
            name: predicate_column,
            column_type: DirectPrimitiveColumnType::Date32,
        });
        profile_columns.extend(payload_columns);
        log_direct_primitive_fold_profile(&path, &profile_columns, &scan_metrics);
        Ok(Some(scan_metrics))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn scan_parquet_i64_lookup_decimal_utf8_selected_primitive_columns_parallel<
        Lookup,
        Predicate,
        Consume,
    >(
        &self,
        path: impl Into<PathBuf>,
        row_groups: Vec<usize>,
        key_column: String,
        quantity_column: (String, DirectPrimitiveColumnType),
        first_utf8_column: String,
        second_utf8_column: String,
        payload_columns: Vec<(String, DirectPrimitiveColumnType)>,
        lookup: Lookup,
        predicate: Predicate,
        consume: Consume,
    ) -> Result<Option<DirectPrimitiveColumnScanMetrics>>
    where
        Lookup: Fn(i64) -> Option<u8> + Sync,
        Predicate: Fn(u8, i64, &[u8], &[u8]) -> bool + Sync,
        Consume: for<'a> Fn(BatchView<'a>) -> Result<()> + Sync,
    {
        let path = path.into();
        let quantity_column = OwnedDirectPrimitiveColumnSpec {
            name: quantity_column.0,
            column_type: quantity_column.1,
        };
        let payload_columns = payload_columns
            .into_iter()
            .map(|(name, column_type)| OwnedDirectPrimitiveColumnSpec { name, column_type })
            .collect::<Vec<_>>();
        let mut scan_metrics = DirectPrimitiveColumnScanMetrics::default();
        if row_groups.is_empty() {
            return Ok(Some(scan_metrics));
        }
        let workers = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(4)
            .min(row_groups.len());
        let (sender, receiver) = mpsc::channel();
        let row_group_partitions = partition_row_groups_balanced(&row_groups, workers);
        std::thread::scope(|scope| {
            for row_group_partition in row_group_partitions {
                let sender = sender.clone();
                let engine = self.clone();
                let path = path.clone();
                let key_column = key_column.clone();
                let quantity_column = quantity_column.clone();
                let first_utf8_column = first_utf8_column.clone();
                let second_utf8_column = second_utf8_column.clone();
                let payload_columns = payload_columns.clone();
                let lookup = &lookup;
                let predicate = &predicate;
                let consume = &consume;
                scope.spawn(move || {
                    let quantity_spec = DirectPrimitiveColumnSpec {
                        name: &quantity_column.name,
                        column_type: quantity_column.column_type,
                    };
                    let payload_specs =
                        OwnedDirectPrimitiveColumnSpec::borrowed_specs(&payload_columns);
                    let result =
                        scan_parquet_i64_lookup_decimal_utf8_selected_primitive_columns_with_store(
                            &path,
                            &row_group_partition,
                            &key_column,
                            quantity_spec,
                            &first_utf8_column,
                            &second_utf8_column,
                            &payload_specs,
                            engine.file_cache.clone(),
                            engine.object_store.as_ref(),
                            lookup,
                            predicate,
                            |columns| consume(BatchView::from_raw_columns(columns)),
                        );
                    let _ = sender.send(result);
                });
            }
        });
        drop(sender);
        for received in receiver {
            let Some(metrics) = received? else {
                return Ok(None);
            };
            scan_metrics.merge_from(metrics);
        }
        let mut profile_columns = Vec::with_capacity(payload_columns.len() + 4);
        profile_columns.push(OwnedDirectPrimitiveColumnSpec {
            name: key_column,
            column_type: DirectPrimitiveColumnType::I64,
        });
        profile_columns.push(quantity_column);
        profile_columns.push(OwnedDirectPrimitiveColumnSpec {
            name: first_utf8_column,
            column_type: DirectPrimitiveColumnType::I32,
        });
        profile_columns.push(OwnedDirectPrimitiveColumnSpec {
            name: second_utf8_column,
            column_type: DirectPrimitiveColumnType::I32,
        });
        profile_columns.extend(payload_columns);
        log_direct_primitive_fold_profile(&path, &profile_columns, &scan_metrics);
        Ok(Some(scan_metrics))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn collect_parquet_i64_by_utf8_dictionary_predicate_parallel<Predicate>(
        &self,
        path: impl Into<PathBuf>,
        row_groups: Vec<usize>,
        key_column: String,
        predicate_column: String,
        max_selected_ratio: (usize, usize),
        predicate: Predicate,
    ) -> Result<Option<(Vec<i64>, DirectPrimitiveColumnScanMetrics)>>
    where
        Predicate: Fn(&[u8]) -> bool + Sync,
    {
        let path = path.into();
        let mut output = Vec::new();
        let mut scan_metrics = DirectPrimitiveColumnScanMetrics::default();
        if row_groups.is_empty() {
            return Ok(Some((output, scan_metrics)));
        }
        let workers = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(4)
            .min(row_groups.len());
        let (sender, receiver) = mpsc::channel();
        let row_group_partitions = partition_row_groups_balanced(&row_groups, workers);
        std::thread::scope(|scope| {
            for row_group_partition in row_group_partitions {
                let sender = sender.clone();
                let engine = self.clone();
                let path = path.clone();
                let key_column = key_column.clone();
                let predicate_column = predicate_column.clone();
                let predicate = &predicate;
                scope.spawn(move || {
                    let result = collect_parquet_i64_by_utf8_dictionary_predicate_with_store(
                        &path,
                        &row_group_partition,
                        &key_column,
                        &predicate_column,
                        max_selected_ratio,
                        engine.file_cache.clone(),
                        engine.object_store.as_ref(),
                        predicate,
                    );
                    let _ = sender.send(result);
                });
            }
        });
        drop(sender);
        for received in receiver {
            let Some((mut rows, metrics)) = received? else {
                return Ok(None);
            };
            output.append(&mut rows);
            scan_metrics.merge_from(metrics);
        }
        log_direct_primitive_named_profile(
            &path,
            &[key_column.as_str(), predicate_column.as_str()],
            &scan_metrics,
        );
        Ok(Some((output, scan_metrics)))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn collect_parquet_i32_predicate_i64_lookup_selected_i64_mapped_parallel<
        Tag,
        Predicate,
        Lookup,
    >(
        &self,
        path: impl Into<PathBuf>,
        row_groups: Vec<usize>,
        predicate_column: (String, DirectPrimitiveColumnType),
        key_column: String,
        payload_column: String,
        max_candidate_ratio: (usize, usize),
        predicate: Predicate,
        lookup: Lookup,
    ) -> Result<Option<(Vec<(i64, Tag)>, DirectPrimitiveColumnScanMetrics)>>
    where
        Tag: Copy + Send,
        Predicate: Fn(i32) -> Option<Tag> + Sync,
        Lookup: Fn(i64) -> bool + Sync,
    {
        let path = path.into();
        let mut output = Vec::new();
        let mut scan_metrics = DirectPrimitiveColumnScanMetrics::default();
        if row_groups.is_empty() {
            return Ok(Some((output, scan_metrics)));
        }
        let workers = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(4)
            .min(row_groups.len());
        let (sender, receiver) = mpsc::channel();
        let row_group_partitions = partition_row_groups_balanced(&row_groups, workers);
        std::thread::scope(|scope| {
            for row_group_partition in row_group_partitions {
                let sender = sender.clone();
                let engine = self.clone();
                let path = path.clone();
                let predicate_column = predicate_column.clone();
                let key_column = key_column.clone();
                let payload_column = payload_column.clone();
                let predicate = &predicate;
                let lookup = &lookup;
                scope.spawn(move || {
                    let predicate_spec = DirectPrimitiveColumnSpec {
                        name: &predicate_column.0,
                        column_type: predicate_column.1,
                    };
                    let result =
                        collect_parquet_i32_predicate_i64_lookup_selected_i64_mapped_with_store(
                            &path,
                            &row_group_partition,
                            predicate_spec,
                            &key_column,
                            &payload_column,
                            max_candidate_ratio,
                            engine.file_cache.clone(),
                            engine.object_store.as_ref(),
                            predicate,
                            lookup,
                        );
                    let _ = sender.send(result);
                });
            }
        });
        drop(sender);
        for received in receiver {
            let Some((mut rows, metrics)) = received? else {
                return Ok(None);
            };
            output.append(&mut rows);
            scan_metrics.merge_from(metrics);
        }
        let profile_columns = vec![
            OwnedDirectPrimitiveColumnSpec {
                name: predicate_column.0,
                column_type: predicate_column.1,
            },
            OwnedDirectPrimitiveColumnSpec {
                name: key_column,
                column_type: DirectPrimitiveColumnType::I64,
            },
            OwnedDirectPrimitiveColumnSpec {
                name: payload_column,
                column_type: DirectPrimitiveColumnType::I64,
            },
        ];
        log_direct_primitive_fold_profile(&path, &profile_columns, &scan_metrics);
        Ok(Some((output, scan_metrics)))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn collect_parquet_i64_two_utf8_i64_mapped_parallel<V, Map>(
        &self,
        path: impl Into<PathBuf>,
        batch_size: usize,
        row_groups: Vec<usize>,
        key_column: String,
        first_utf8_column: String,
        second_utf8_column: String,
        numeric_column: String,
        map: Map,
    ) -> Result<Option<(Vec<(i64, V)>, DirectPrimitiveColumnScanMetrics)>>
    where
        V: Copy + Send,
        Map: Fn(&[u8], &[u8], i64) -> Option<V> + Sync,
    {
        let path = path.into();
        let mut output = Vec::new();
        let mut scan_metrics = DirectPrimitiveColumnScanMetrics::default();
        if row_groups.is_empty() {
            return Ok(Some((output, scan_metrics)));
        }
        let workers = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(4)
            .min(row_groups.len());
        let (sender, receiver) = mpsc::channel();
        let row_group_partitions = partition_row_groups_balanced(&row_groups, workers);
        std::thread::scope(|scope| {
            for row_group_partition in row_group_partitions {
                let sender = sender.clone();
                let engine = self.clone();
                let path = path.clone();
                let key_column = key_column.clone();
                let first_utf8_column = first_utf8_column.clone();
                let second_utf8_column = second_utf8_column.clone();
                let numeric_column = numeric_column.clone();
                let map = &map;
                scope.spawn(move || {
                    let result = collect_parquet_i64_two_utf8_i64_mapped_with_store(
                        &path,
                        batch_size,
                        &row_group_partition,
                        &key_column,
                        &first_utf8_column,
                        &second_utf8_column,
                        &numeric_column,
                        engine.file_cache.clone(),
                        engine.object_store.as_ref(),
                        map,
                    );
                    let _ = sender.send(result);
                });
            }
        });
        drop(sender);
        for received in receiver {
            let Some((mut rows, metrics)) = received? else {
                return Ok(None);
            };
            output.append(&mut rows);
            scan_metrics.merge_from(metrics);
        }
        let profile_columns = vec![
            OwnedDirectPrimitiveColumnSpec {
                name: key_column,
                column_type: DirectPrimitiveColumnType::I64,
            },
            OwnedDirectPrimitiveColumnSpec {
                name: first_utf8_column,
                column_type: DirectPrimitiveColumnType::I32,
            },
            OwnedDirectPrimitiveColumnSpec {
                name: second_utf8_column,
                column_type: DirectPrimitiveColumnType::I32,
            },
            OwnedDirectPrimitiveColumnSpec {
                name: numeric_column,
                column_type: DirectPrimitiveColumnType::I64,
            },
        ];
        log_direct_primitive_fold_profile(&path, &profile_columns, &scan_metrics);
        Ok(Some((output, scan_metrics)))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn scan_parquet_i64_lookup_staged_two_i64_selected_primitive_columns_parallel<
        FirstTag,
        FinalTag,
        Lookup,
        FirstPredicate,
        SecondPredicate,
        Consume,
    >(
        &self,
        path: impl Into<PathBuf>,
        row_groups: Vec<usize>,
        key_column: String,
        first_predicate_column: String,
        second_predicate_column: String,
        payload_columns: Vec<(String, DirectPrimitiveColumnType)>,
        max_candidate_ratio: (usize, usize),
        lookup: Lookup,
        first_predicate: FirstPredicate,
        second_predicate: SecondPredicate,
        consume: Consume,
    ) -> Result<Option<DirectPrimitiveColumnScanMetrics>>
    where
        FirstTag: Copy + Send + Sync,
        FinalTag: Copy + Send + Sync,
        Lookup: Fn(i64) -> bool + Sync,
        FirstPredicate: Fn(i64) -> Option<FirstTag> + Sync,
        SecondPredicate: Fn(FirstTag, i64) -> Option<FinalTag> + Sync,
        Consume: for<'a> Fn(&[FinalTag], BatchView<'a>) -> Result<()> + Sync,
    {
        let path = path.into();
        let payload_columns = payload_columns
            .into_iter()
            .map(|(name, column_type)| OwnedDirectPrimitiveColumnSpec { name, column_type })
            .collect::<Vec<_>>();
        let mut scan_metrics = DirectPrimitiveColumnScanMetrics::default();
        if row_groups.is_empty() {
            return Ok(Some(scan_metrics));
        }
        let workers = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(4)
            .min(row_groups.len());
        let (sender, receiver) = mpsc::channel();
        let row_group_partitions = partition_row_groups_balanced(&row_groups, workers);
        std::thread::scope(|scope| {
            for (partition_index, row_group_partition) in
                row_group_partitions.into_iter().enumerate()
            {
                let sender = sender.clone();
                let engine = self.clone();
                let path = path.clone();
                let key_column = key_column.clone();
                let first_predicate_column = first_predicate_column.clone();
                let second_predicate_column = second_predicate_column.clone();
                let payload_columns = payload_columns.clone();
                let lookup = &lookup;
                let first_predicate = &first_predicate;
                let second_predicate = &second_predicate;
                let consume = &consume;
                scope.spawn(move || {
                    let partition_profile = direct_primitive_profile_enabled();
                    let partition_started = partition_profile.then(Instant::now);
                    let partition_row_groups = row_group_partition.len();
                    let payload_specs =
                        OwnedDirectPrimitiveColumnSpec::borrowed_specs(&payload_columns);
                    let result = scan_parquet_i64_lookup_staged_two_i64_selected_primitive_columns_with_store(
                        &path,
                        &row_group_partition,
                        &key_column,
                        &first_predicate_column,
                        &second_predicate_column,
                        &payload_specs,
                        max_candidate_ratio,
                        engine.file_cache.clone(),
                        engine.object_store.as_ref(),
                        lookup,
                        first_predicate,
                        second_predicate,
                        |tags, columns| consume(tags, BatchView::from_raw_columns(columns)),
                    );
                    let elapsed = partition_started
                        .map(|started| elapsed_nanos(started.elapsed()))
                        .unwrap_or_default();
                    let _ = sender.send((partition_index, partition_row_groups, elapsed, result));
                });
            }
        });
        drop(sender);
        for (partition_index, partition_row_groups, elapsed, received) in receiver {
            if direct_primitive_profile_enabled() {
                eprintln!(
                    "[dodam:direct-primitive-partition-profile] partition={} row_groups={} elapsed={:.3} ms",
                    partition_index,
                    partition_row_groups,
                    nanos_to_millis(elapsed),
                );
            }
            let Some(metrics) = received? else {
                return Ok(None);
            };
            scan_metrics.merge_from(metrics);
        }
        let mut profile_columns = Vec::with_capacity(payload_columns.len() + 3);
        profile_columns.push(OwnedDirectPrimitiveColumnSpec {
            name: key_column,
            column_type: DirectPrimitiveColumnType::I64,
        });
        profile_columns.push(OwnedDirectPrimitiveColumnSpec {
            name: first_predicate_column,
            column_type: DirectPrimitiveColumnType::I64,
        });
        profile_columns.push(OwnedDirectPrimitiveColumnSpec {
            name: second_predicate_column,
            column_type: DirectPrimitiveColumnType::I64,
        });
        profile_columns.extend(payload_columns);
        log_direct_primitive_fold_profile(&path, &profile_columns, &scan_metrics);
        Ok(Some(scan_metrics))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn scan_parquet_required_plain_primitive_in_list_desc<F>(
        &self,
        path: impl AsRef<Path>,
        batch_size: usize,
        row_groups: &[usize],
        columns: &[DirectPrimitiveColumnSpec<'_>],
        filter_index: usize,
        filter_i32_values: &[i32],
        filter_i64_values: &[i64],
        consume: F,
    ) -> Result<Option<DirectPrimitiveColumnScanMetrics>>
    where
        F: FnMut(DirectOrderedPrimitiveBatch) -> Result<()>,
    {
        scan_parquet_required_plain_primitive_in_list_desc_with_store(
            path.as_ref(),
            batch_size,
            row_groups,
            columns,
            filter_index,
            filter_i32_values,
            filter_i64_values,
            self.file_cache.clone(),
            self.object_store.as_ref(),
            consume,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn scan_parquet_required_plain_primitive_in_list_desc_selected_pages<F>(
        &self,
        path: impl AsRef<Path>,
        batch_size: usize,
        row_groups: &[usize],
        columns: &[DirectPrimitiveColumnSpec<'_>],
        filter_index: usize,
        filter_i32_values: &[i32],
        filter_i64_values: &[i64],
        consume: F,
    ) -> Result<Option<DirectPrimitiveColumnScanMetrics>>
    where
        F: for<'a> FnMut(DirectSelectedPrimitivePageBatch<'a>) -> Result<()>,
    {
        scan_parquet_required_plain_primitive_in_list_desc_selected_pages_with_store(
            path.as_ref(),
            batch_size,
            row_groups,
            columns,
            filter_index,
            filter_i32_values,
            filter_i64_values,
            self.file_cache.clone(),
            self.object_store.as_ref(),
            consume,
        )
    }

    pub(crate) fn scan_parquet_primitive_columns_parallel_view_fold<S, Init, Consume, Merge>(
        &self,
        path: impl Into<PathBuf>,
        batch_size: usize,
        row_groups: Vec<usize>,
        columns: Vec<(String, DirectPrimitiveColumnType)>,
        init: Init,
        consume: Consume,
        merge: Merge,
    ) -> Result<Option<(S, DirectPrimitiveColumnScanMetrics)>>
    where
        S: Send,
        Init: Fn() -> S + Sync,
        Consume: for<'a> Fn(&mut S, BatchView<'a>) -> Result<()> + Sync,
        Merge: Fn(&mut S, S) -> Result<()> + Sync,
    {
        let columns = columns
            .into_iter()
            .map(|(name, column_type)| OwnedDirectPrimitiveColumnSpec { name, column_type })
            .collect();
        self.scan_parquet_primitive_columns_parallel_fold(
            DirectPrimitiveFoldPlan::new(path, batch_size, row_groups, columns),
            init,
            consume,
            merge,
        )
    }

    pub(crate) fn scan_parquet_primitive_columns_parallel_view_fold_with_location<
        S,
        Init,
        Consume,
        Merge,
    >(
        &self,
        path: impl Into<PathBuf>,
        batch_size: usize,
        row_groups: Vec<usize>,
        columns: Vec<(String, DirectPrimitiveColumnType)>,
        init: Init,
        consume: Consume,
        merge: Merge,
    ) -> Result<Option<(S, DirectPrimitiveColumnScanMetrics)>>
    where
        S: Send,
        Init: Fn() -> S + Sync,
        Consume:
            for<'a> Fn(&mut S, DirectPrimitiveBatchLocation, BatchView<'a>) -> Result<()> + Sync,
        Merge: Fn(&mut S, S) -> Result<()> + Sync,
    {
        let columns = columns
            .into_iter()
            .map(|(name, column_type)| OwnedDirectPrimitiveColumnSpec { name, column_type })
            .collect();
        self.scan_parquet_primitive_columns_parallel_fold_with_location(
            DirectPrimitiveFoldPlan::new(path, batch_size, row_groups, columns),
            init,
            consume,
            merge,
        )
    }

    fn scan_parquet_primitive_columns_parallel_fold_with_location<S, Init, Consume, Merge>(
        &self,
        plan: DirectPrimitiveFoldPlan,
        init: Init,
        consume: Consume,
        merge: Merge,
    ) -> Result<Option<(S, DirectPrimitiveColumnScanMetrics)>>
    where
        S: Send,
        Init: Fn() -> S + Sync,
        Consume:
            for<'a> Fn(&mut S, DirectPrimitiveBatchLocation, BatchView<'a>) -> Result<()> + Sync,
        Merge: Fn(&mut S, S) -> Result<()> + Sync,
    {
        let mut state = init();
        let mut scan_metrics = DirectPrimitiveColumnScanMetrics::default();
        if plan.row_groups.is_empty() {
            return Ok(Some((state, scan_metrics)));
        }

        let workers = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(4)
            .min(plan.row_groups.len());
        let (sender, receiver) = mpsc::channel();
        let row_group_partitions = partition_row_groups_balanced(&plan.row_groups, workers);
        std::thread::scope(|scope| {
            for row_group_partition in row_group_partitions {
                let sender = sender.clone();
                let engine = self.clone();
                let path = plan.path.clone();
                let columns = plan.columns.clone();
                let init = &init;
                let consume = &consume;
                scope.spawn(move || {
                    let specs = OwnedDirectPrimitiveColumnSpec::borrowed_specs(&columns);
                    let mut state = init();
                    let mut metrics = DirectPrimitiveColumnScanMetrics::default();
                    for row_group in row_group_partition {
                        let mut row_offset = 0usize;
                        let result = engine.scan_parquet_primitive_columns_view(
                            &path,
                            plan.batch_size,
                            &[row_group],
                            &specs,
                            |batch| {
                                let location = DirectPrimitiveBatchLocation {
                                    row_group,
                                    row_offset,
                                };
                                row_offset = row_offset.saturating_add(batch.num_rows());
                                consume(&mut state, location, batch)
                            },
                        );
                        match result {
                            Ok(Some(partial_metrics)) => metrics.merge_from(partial_metrics),
                            Ok(None) => {
                                let _ = sender.send(Ok(None));
                                return;
                            }
                            Err(error) => {
                                let _ = sender.send(Err(error));
                                return;
                            }
                        }
                    }
                    let _ = sender.send(Ok(Some((state, metrics))));
                });
            }
        });
        drop(sender);

        for received in receiver {
            let Some((partial, metrics)) = received? else {
                return Ok(None);
            };
            merge(&mut state, partial)?;
            scan_metrics.merge_from(metrics);
        }
        log_direct_primitive_fold_profile(&plan.path, &plan.columns, &scan_metrics);
        Ok(Some((state, scan_metrics)))
    }

    fn scan_parquet_primitive_columns_parallel_fold<S, Init, Consume, Merge>(
        &self,
        plan: DirectPrimitiveFoldPlan,
        init: Init,
        consume: Consume,
        merge: Merge,
    ) -> Result<Option<(S, DirectPrimitiveColumnScanMetrics)>>
    where
        S: Send,
        Init: Fn() -> S + Sync,
        Consume: for<'a> Fn(&mut S, BatchView<'a>) -> Result<()> + Sync,
        Merge: Fn(&mut S, S) -> Result<()> + Sync,
    {
        let mut state = init();
        let mut scan_metrics = DirectPrimitiveColumnScanMetrics::default();
        if plan.row_groups.is_empty() {
            return Ok(Some((state, scan_metrics)));
        }
        if plan.row_groups.len() <= 1 {
            let specs = OwnedDirectPrimitiveColumnSpec::borrowed_specs(&plan.columns);
            let Some(metrics) = self.scan_parquet_primitive_columns_view(
                &plan.path,
                plan.batch_size,
                &plan.row_groups,
                &specs,
                |batch| consume(&mut state, batch),
            )?
            else {
                return Ok(None);
            };
            log_direct_primitive_fold_profile(&plan.path, &plan.columns, &metrics);
            return Ok(Some((state, metrics)));
        }

        let workers = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(4)
            .min(plan.row_groups.len());
        let (sender, receiver) = mpsc::channel();
        let row_group_partitions = partition_row_groups_balanced(&plan.row_groups, workers);
        std::thread::scope(|scope| {
            for row_group_partition in row_group_partitions {
                let sender = sender.clone();
                let engine = self.clone();
                let path = plan.path.clone();
                let columns = plan.columns.clone();
                let row_groups = row_group_partition;
                let init = &init;
                let consume = &consume;
                scope.spawn(move || {
                    let specs = OwnedDirectPrimitiveColumnSpec::borrowed_specs(&columns);
                    let mut state = init();
                    let result = engine.scan_parquet_primitive_columns_view(
                        &path,
                        plan.batch_size,
                        &row_groups,
                        &specs,
                        |batch| consume(&mut state, batch),
                    );
                    let _ =
                        sender.send(result.map(|metrics| metrics.map(|metrics| (state, metrics))));
                });
            }
        });
        drop(sender);

        for received in receiver {
            let Some((partial, metrics)) = received? else {
                return Ok(None);
            };
            merge(&mut state, partial)?;
            scan_metrics.merge_from(metrics);
        }
        log_direct_primitive_fold_profile(&plan.path, &plan.columns, &scan_metrics);
        Ok(Some((state, scan_metrics)))
    }

    #[allow(clippy::too_many_arguments)]
    fn scan_parquet_i32_i64_decimal_i32_selected_batch_fold<S, Init, Consume, Merge>(
        &self,
        path: impl AsRef<Path>,
        batch_size: usize,
        row_groups: &[usize],
        columns: [&str; 4],
        decimal_precision: u8,
        decimal_scale: i8,
        filter: DecimalDateRangeFilter,
        init: Init,
        consume: Consume,
        merge: Merge,
    ) -> Result<Option<(S, DirectPrimitiveColumnScanMetrics)>>
    where
        S: Send,
        Init: Fn() -> S + Sync,
        Consume: for<'a> Fn(&mut S, BatchView<'a>) -> Result<()> + Sync,
        Merge: Fn(&mut S, S) -> Result<()> + Sync,
    {
        let decimal_min = option_i128_to_i64(filter.decimal_min)?;
        let decimal_max = option_i128_to_i64(filter.decimal_max)?;
        let path = path.as_ref();
        let mut state = init();
        let mut scan_metrics = DirectPrimitiveColumnScanMetrics::default();
        if row_groups.is_empty() {
            return Ok(Some((state, scan_metrics)));
        }
        let workers = fused_dictionary_selected_workers(row_groups.len());
        let (sender, receiver) = mpsc::channel();
        let row_group_partitions = partition_row_groups_balanced(row_groups, workers);
        std::thread::scope(|scope| {
            for row_group_partition in row_group_partitions {
                let sender = sender.clone();
                let engine = self.clone();
                let path = path.to_path_buf();
                let row_groups = row_group_partition;
                let init = &init;
                let consume = &consume;
                scope.spawn(move || {
                    let mut state = init();
                    let result = scan_parquet_i32_i64_decimal_i32_selected_with_store(
                        &path,
                        batch_size,
                        &row_groups,
                        columns,
                        decimal_precision,
                        decimal_scale,
                        decimal_min,
                        decimal_max,
                        filter.date_min,
                        filter.date_max,
                        engine.file_cache.clone(),
                        engine.object_store.as_ref(),
                        |columns| consume(&mut state, BatchView::from_raw_columns(columns)),
                    );
                    let _ =
                        sender.send(result.map(|metrics| metrics.map(|metrics| (state, metrics))));
                });
            }
        });
        drop(sender);
        for received in receiver {
            let Some((partial, metrics)) = received? else {
                return Ok(None);
            };
            merge(&mut state, partial)?;
            scan_metrics.merge_from(metrics);
        }
        let profile_columns = columns
            .iter()
            .map(|name| OwnedDirectPrimitiveColumnSpec {
                name: (*name).to_string(),
                column_type: DirectPrimitiveColumnType::I64,
            })
            .collect::<Vec<_>>();
        log_direct_primitive_fold_profile(path, &profile_columns, &scan_metrics);
        Ok(Some((state, scan_metrics)))
    }

    #[allow(clippy::too_many_arguments)]
    fn scan_parquet_i32_i64_decimal_i32_selected_typed_fold<S, Init, Consume, Merge>(
        &self,
        path: impl AsRef<Path>,
        batch_size: usize,
        row_groups: &[usize],
        columns: [&str; 4],
        decimal_precision: u8,
        decimal_scale: i8,
        filter: DecimalDateRangeFilter,
        init: Init,
        consume: Consume,
        merge: Merge,
    ) -> Result<Option<(S, DirectPrimitiveColumnScanMetrics)>>
    where
        S: Send,
        Init: Fn() -> S + Sync,
        Consume: for<'a> Fn(&mut S, DirectI32I64DecimalI32SelectedBatch<'a>) -> Result<()> + Sync,
        Merge: Fn(&mut S, S) -> Result<()> + Sync,
    {
        let decimal_min = option_i128_to_i64(filter.decimal_min)?;
        let decimal_max = option_i128_to_i64(filter.decimal_max)?;
        let path = path.as_ref();
        let mut state = init();
        let mut scan_metrics = DirectPrimitiveColumnScanMetrics::default();
        if row_groups.is_empty() {
            return Ok(Some((state, scan_metrics)));
        }
        let workers = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(4)
            .min(row_groups.len());
        let (sender, receiver) = mpsc::channel();
        let row_group_partitions = partition_row_groups_balanced(row_groups, workers);
        std::thread::scope(|scope| {
            for row_group_partition in row_group_partitions {
                let sender = sender.clone();
                let engine = self.clone();
                let path = path.to_path_buf();
                let row_groups = row_group_partition;
                let init = &init;
                let consume = &consume;
                scope.spawn(move || {
                    let mut state = init();
                    let result = scan_parquet_i32_i64_decimal_i32_selected_typed_with_store(
                        &path,
                        batch_size,
                        &row_groups,
                        columns,
                        decimal_precision,
                        decimal_scale,
                        decimal_min,
                        decimal_max,
                        filter.date_min,
                        filter.date_max,
                        engine.file_cache.clone(),
                        engine.object_store.as_ref(),
                        |batch| consume(&mut state, batch),
                    );
                    let _ =
                        sender.send(result.map(|metrics| metrics.map(|metrics| (state, metrics))));
                });
            }
        });
        drop(sender);
        for received in receiver {
            let Some((partial, metrics)) = received? else {
                return Ok(None);
            };
            merge(&mut state, partial)?;
            scan_metrics.merge_from(metrics);
        }
        let profile_columns = columns
            .iter()
            .map(|name| OwnedDirectPrimitiveColumnSpec {
                name: (*name).to_string(),
                column_type: DirectPrimitiveColumnType::I64,
            })
            .collect::<Vec<_>>();
        log_direct_primitive_fold_profile(path, &profile_columns, &scan_metrics);
        Ok(Some((state, scan_metrics)))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn scan_parquet_i32_i32_dictionary_i64_decimal_selected_fold<
        S,
        Init,
        Consume,
        Merge,
    >(
        &self,
        path: impl AsRef<Path>,
        batch_size: usize,
        row_groups: &[usize],
        columns: [&str; 5],
        fallback: &[u8],
        decimal_min: Option<i64>,
        decimal_max: Option<i64>,
        workers_override: Option<usize>,
        init: Init,
        consume: Consume,
        merge: Merge,
    ) -> Result<Option<(S, DirectPrimitiveColumnScanMetrics)>>
    where
        S: Send,
        Init: Fn() -> S + Sync,
        Consume:
            for<'a> Fn(&mut S, DirectI32I32DictionaryI64SelectedBatch<'a>) -> Result<()> + Sync,
        Merge: Fn(&mut S, S) -> Result<()> + Sync,
    {
        let path = path.as_ref();
        let mut state = init();
        let mut scan_metrics = DirectPrimitiveColumnScanMetrics::default();
        if row_groups.is_empty() {
            return Ok(Some((state, scan_metrics)));
        }
        let workers = workers_override
            .filter(|workers| *workers > 0)
            .unwrap_or_else(|| fused_dictionary_selected_workers(row_groups.len()))
            .min(row_groups.len().max(1));
        let (sender, receiver) = mpsc::channel();
        let row_group_partitions = partition_row_groups_balanced(row_groups, workers);
        std::thread::scope(|scope| {
            for row_group_partition in row_group_partitions {
                let sender = sender.clone();
                let engine = self.clone();
                let path = path.to_path_buf();
                let row_groups = row_group_partition;
                let fallback = fallback.to_vec();
                let init = &init;
                let consume = &consume;
                scope.spawn(move || {
                    let mut state = init();
                    let result =
                        scan_parquet_i32_i32_dictionary_i64_decimal_selected_typed_with_store(
                            &path,
                            batch_size,
                            &row_groups,
                            columns,
                            &fallback,
                            decimal_min,
                            decimal_max,
                            engine.file_cache.clone(),
                            engine.object_store.as_ref(),
                            |batch| consume(&mut state, batch),
                        );
                    let _ =
                        sender.send(result.map(|metrics| metrics.map(|metrics| (state, metrics))));
                });
            }
        });
        drop(sender);
        for received in receiver {
            let Some((partial, metrics)) = received? else {
                return Ok(None);
            };
            merge(&mut state, partial)?;
            scan_metrics.merge_from(metrics);
        }
        let profile_columns = columns
            .iter()
            .map(|name| OwnedDirectPrimitiveColumnSpec {
                name: (*name).to_string(),
                column_type: DirectPrimitiveColumnType::I64,
            })
            .collect::<Vec<_>>();
        log_direct_primitive_fold_profile(path, &profile_columns, &scan_metrics);
        Ok(Some((state, scan_metrics)))
    }

    pub(crate) fn scan_parquet_i32_i64_byte_array_columns<F>(
        &self,
        path: impl AsRef<Path>,
        batch_size: usize,
        row_groups: &[usize],
        columns: [&str; 3],
        consume: F,
    ) -> Result<Option<DirectColumnScanMetrics>>
    where
        F: FnMut(&[i32], &[i64], &[i16], &[parquet::data_type::ByteArray]) -> Result<Option<()>>,
    {
        scan_parquet_i32_i64_byte_array_columns_with_store(
            path.as_ref(),
            batch_size,
            row_groups,
            columns,
            self.file_cache.clone(),
            self.object_store.as_ref(),
            consume,
        )
    }

    pub(crate) fn scan_parquet_i32_byte_array_columns<F>(
        &self,
        path: impl AsRef<Path>,
        batch_size: usize,
        row_groups: &[usize],
        columns: [&str; 2],
        consume: F,
    ) -> Result<Option<DirectColumnScanMetrics>>
    where
        F: FnMut(&[i32], &[i16], &[parquet::data_type::ByteArray]) -> Result<Option<()>>,
    {
        let path = path.as_ref();
        let metrics = scan_parquet_i32_byte_array_columns_with_store(
            path,
            batch_size,
            row_groups,
            columns,
            self.file_cache.clone(),
            self.object_store.as_ref(),
            consume,
        )?;
        if let Some(metrics) = metrics.as_ref() {
            log_direct_column_scan_profile(path, &columns, "i32-bytearray", metrics);
        }
        Ok(metrics)
    }

    pub(crate) fn scan_parquet_i32_i32_columns<F>(
        &self,
        path: impl AsRef<Path>,
        batch_size: usize,
        row_groups: &[usize],
        columns: [&str; 2],
        consume: F,
    ) -> Result<Option<DirectColumnScanMetrics>>
    where
        F: FnMut(&[i32], Option<&[i16]>, &[i32], Option<&[i16]>) -> Result<Option<()>>,
    {
        let path = path.as_ref();
        let metrics = scan_parquet_i32_i32_columns_with_store(
            path,
            batch_size,
            row_groups,
            columns,
            self.file_cache.clone(),
            self.object_store.as_ref(),
            consume,
        )?;
        if let Some(metrics) = metrics.as_ref() {
            log_direct_column_scan_profile(path, &columns, "i32-i32", metrics);
        }
        Ok(metrics)
    }

    pub(crate) fn scan_parquet_i32_byte_array_selected_by_i32<P, F>(
        &self,
        path: impl AsRef<Path>,
        row_groups: &[usize],
        columns: [&str; 2],
        predicate: P,
        consume: F,
    ) -> Result<Option<DirectColumnScanMetrics>>
    where
        P: Fn(i32) -> bool,
        F: FnMut(&[i32], &[i32], &[bytes::Bytes]) -> Result<Option<()>>,
    {
        let path = path.as_ref();
        let metrics = scan_parquet_i32_byte_array_selected_by_i32_with_store(
            path,
            row_groups,
            columns,
            self.file_cache.clone(),
            self.object_store.as_ref(),
            predicate,
            consume,
        )?;
        if let Some(metrics) = metrics.as_ref() {
            log_direct_column_scan_profile(
                path,
                &columns,
                "i32-bytearray-selected-by-i32",
                metrics,
            );
        }
        Ok(metrics)
    }

    pub(crate) fn scan_parquet_i32_selected_by_byte_array_prefix<F>(
        &self,
        path: impl AsRef<Path>,
        row_groups: &[usize],
        columns: [&str; 2],
        prefix: &[u8],
        consume: F,
    ) -> Result<Option<DirectColumnScanMetrics>>
    where
        F: FnMut(&[i32], &[i32], &[bytes::Bytes]) -> Result<Option<()>>,
    {
        let path = path.as_ref();
        let metrics = scan_parquet_i32_selected_by_byte_array_prefix_with_store(
            path,
            row_groups,
            columns,
            prefix,
            self.file_cache.clone(),
            self.object_store.as_ref(),
            consume,
        )?;
        if let Some(metrics) = metrics.as_ref() {
            log_direct_column_scan_profile(
                path,
                &columns,
                "i32-selected-by-bytearray-prefix",
                metrics,
            );
        }
        Ok(metrics)
    }

    pub(crate) fn scan_parquet_i32_i64_dictionary_id_columns<F>(
        &self,
        path: &Path,
        batch_size: usize,
        row_groups: &[usize],
        columns: [&str; 3],
        consume: F,
    ) -> Result<Option<DirectColumnScanMetrics>>
    where
        F: FnMut(&[i32], &[i64], Option<&[i16]>, &[i32], &[bytes::Bytes]) -> Result<Option<()>>,
    {
        let metrics = scan_parquet_i32_i64_dictionary_id_columns_with_store(
            path,
            batch_size,
            row_groups,
            columns,
            self.file_cache.clone(),
            self.object_store.as_ref(),
            consume,
        )?;
        if let Some(metrics) = metrics {
            log_direct_column_scan_profile(path, &columns, "i32-dictionary-id", &metrics);
            return Ok(Some(metrics));
        }
        Ok(None)
    }

    pub(crate) fn scan_parquet_i32_dictionary_id_columns<F>(
        &self,
        path: &Path,
        batch_size: usize,
        row_groups: &[usize],
        columns: [&str; 2],
        consume: F,
    ) -> Result<Option<DirectColumnScanMetrics>>
    where
        F: FnMut(&[i32], Option<&[i16]>, &[i32], &[bytes::Bytes]) -> Result<Option<()>>,
    {
        let metrics = scan_parquet_i32_dictionary_id_columns_with_store(
            path,
            batch_size,
            row_groups,
            columns,
            self.file_cache.clone(),
            self.object_store.as_ref(),
            consume,
        )?;
        if let Some(metrics) = metrics {
            log_direct_column_scan_profile(path, &columns, "i32-dictionary-id", &metrics);
            return Ok(Some(metrics));
        }
        Ok(None)
    }

    pub(crate) fn scan_parquet_i32_dictionary_id_columns_raw<F>(
        &self,
        path: &Path,
        batch_size: usize,
        row_groups: &[usize],
        columns: [&str; 2],
        consume: F,
    ) -> Result<Option<DirectColumnScanMetrics>>
    where
        F: FnMut(&[u8], usize, Option<&[i16]>, &[i32], &[bytes::Bytes]) -> Result<Option<()>>,
    {
        let metrics = scan_parquet_i32_dictionary_id_columns_raw_with_store(
            path,
            batch_size,
            row_groups,
            columns,
            self.file_cache.clone(),
            self.object_store.as_ref(),
            consume,
        )?;
        if let Some(metrics) = metrics {
            log_direct_column_scan_profile(path, &columns, "i32-dictionary-id-raw", &metrics);
            return Ok(Some(metrics));
        }
        Ok(None)
    }

    pub async fn scan_parquet(
        &self,
        path: PathBuf,
        batch_size: usize,
        limit: Option<usize>,
        projection: Projection,
        filter: Option<FilterExpr>,
    ) -> Result<ScanMetrics> {
        let source = self.plan_table_source(path).await?;
        self.scan_table(source, batch_size, limit, projection, filter, None)
            .await
    }

    pub async fn estimate_parquet_projection_compressed_bytes(
        &self,
        path: PathBuf,
        projection: &Projection,
    ) -> Result<u64> {
        let path = self.resolve_table_path(path)?;
        read_parquet_projection_compressed_bytes(
            path,
            projection,
            &self.metadata_cache,
            self.object_store.as_ref(),
        )
    }

    pub async fn scan_table(
        &self,
        source: TableScanSource,
        batch_size: usize,
        limit: Option<usize>,
        projection: Projection,
        filter: Option<FilterExpr>,
        order_by: Option<SortKey>,
    ) -> Result<ScanMetrics> {
        let plan = self.plan_table_source_scan(
            source,
            batch_size,
            limit,
            projection.clone(),
            filter,
            order_by,
        )?;
        let fragment_count = plan.source.fragments.len();
        let stream = self.execute_scan_plan(plan)?;
        collect_metrics(stream, fragment_count, &projection)
    }

    pub async fn scan_parquet_ordered(
        &self,
        path: PathBuf,
        batch_size: usize,
        limit: Option<usize>,
        projection: Projection,
        filter: Option<FilterExpr>,
        order_by: SortExpr,
    ) -> Result<ScanMetrics> {
        self.scan_parquet_ordered_by(
            path,
            batch_size,
            limit,
            projection,
            filter,
            SortKey::from(order_by),
        )
        .await
    }

    pub async fn scan_parquet_ordered_by(
        &self,
        path: PathBuf,
        batch_size: usize,
        limit: Option<usize>,
        projection: Projection,
        filter: Option<FilterExpr>,
        order_by: SortKey,
    ) -> Result<ScanMetrics> {
        let plan = self
            .plan_parquet_scan(
                path,
                batch_size,
                limit,
                projection.clone(),
                filter,
                Some(order_by),
            )
            .await?;
        let fragment_count = plan.source.fragments.len();
        let stream = self.execute_scan_plan(plan)?;
        collect_metrics(stream, fragment_count, &projection)
    }

    pub async fn scan_parquet_batches(
        &self,
        path: PathBuf,
        batch_size: usize,
        limit: Option<usize>,
        projection: Projection,
        filter: Option<FilterExpr>,
    ) -> Result<SendableBatchStream> {
        let source = self.plan_table_source(path).await?;
        self.scan_table_source_batches(source, batch_size, limit, projection, filter, None)
    }

    pub(crate) async fn scan_parquet_batches_fold_view<S, C, F, O>(
        &self,
        path: PathBuf,
        batch_size: usize,
        limit: Option<usize>,
        projection: Projection,
        filter: Option<FilterExpr>,
        mut state: S,
        mut consume: C,
        finish: F,
    ) -> Result<O>
    where
        C: for<'a> FnMut(BatchView<'a>, &mut S) -> Result<()>,
        F: FnOnce(S) -> Result<O>,
    {
        let source = self.plan_table_source(path).await?;
        let plan =
            self.plan_table_source_scan(source, batch_size, limit, projection, filter, None)?;
        self.build_physical_scan_plan(plan)
            .for_each_batch(&mut |batch| {
                if batch.num_rows() > 0 {
                    consume(BatchView::new(batch), &mut state)?;
                }
                Ok(())
            })?;
        finish(state)
    }

    pub async fn scan_parquet_batches_pruned(
        &self,
        path: PathBuf,
        batch_size: usize,
        projection: Projection,
        pruning_predicates: Vec<Expr>,
    ) -> Result<SendableBatchStream> {
        let source = self.plan_table_source(path).await?;
        let estimated_bytes = source.statistics.compressed_bytes;
        let plan = ScanPlan {
            source,
            batch_size,
            limit: None,
            output_projection: projection.clone(),
            scan_projection: projection,
            filter: None,
            residual_filter: None,
            pushdown_predicates: pruning_predicates,
            row_filter_predicates: Vec::new(),
            has_filter: false,
            distinct: false,
            order_by: None,
            estimated_bytes,
            operators: vec![ScanOperator::Scan],
            preserve_order: false,
        };
        self.execute_scan_plan(plan)
    }

    pub async fn scan_parquet_batches_row_filtered(
        &self,
        path: PathBuf,
        batch_size: usize,
        projection: Projection,
        predicates: Vec<Expr>,
    ) -> Result<SendableBatchStream> {
        let source = self.plan_table_source(path).await?;
        let estimated_bytes = source.statistics.compressed_bytes;
        let plan = ScanPlan {
            source,
            batch_size,
            limit: None,
            output_projection: projection.clone(),
            scan_projection: projection,
            filter: None,
            residual_filter: None,
            pushdown_predicates: predicates.clone(),
            row_filter_predicates: predicates,
            has_filter: true,
            distinct: false,
            order_by: None,
            estimated_bytes,
            operators: vec![ScanOperator::Scan],
            preserve_order: false,
        };
        self.execute_scan_plan(plan)
    }

    pub async fn scan_parquet_batches_row_filtered_preserve_order(
        &self,
        path: PathBuf,
        batch_size: usize,
        projection: Projection,
        predicates: Vec<Expr>,
    ) -> Result<SendableBatchStream> {
        let source = self.plan_table_source(path).await?;
        let estimated_bytes = source.statistics.compressed_bytes;
        let plan = ScanPlan {
            source,
            batch_size,
            limit: None,
            output_projection: projection.clone(),
            scan_projection: projection,
            filter: None,
            residual_filter: None,
            pushdown_predicates: predicates.clone(),
            row_filter_predicates: predicates,
            has_filter: true,
            distinct: false,
            order_by: None,
            estimated_bytes,
            operators: vec![ScanOperator::Scan],
            preserve_order: true,
        };
        self.execute_scan_plan(plan)
    }

    pub async fn scan_parquet_batches_preserve_order(
        &self,
        path: PathBuf,
        batch_size: usize,
        projection: Projection,
    ) -> Result<SendableBatchStream> {
        let source = self.plan_table_source(path).await?;
        let estimated_bytes = source.statistics.compressed_bytes;
        let plan = ScanPlan {
            source,
            batch_size,
            limit: None,
            output_projection: projection.clone(),
            scan_projection: projection,
            filter: None,
            residual_filter: None,
            pushdown_predicates: Vec::new(),
            row_filter_predicates: Vec::new(),
            has_filter: false,
            distinct: false,
            order_by: None,
            estimated_bytes,
            operators: vec![ScanOperator::Scan],
            preserve_order: true,
        };
        self.execute_scan_plan(plan)
    }

    pub async fn scan_parquet_filtered_batches_preserve_order(
        &self,
        path: PathBuf,
        batch_size: usize,
        projection: Projection,
        filter: Option<FilterExpr>,
    ) -> Result<SendableBatchStream> {
        let source = self.plan_table_source(path).await?;
        let mut plan =
            self.plan_table_source_scan(source, batch_size, None, projection, filter, None)?;
        plan.preserve_order = true;
        self.execute_scan_plan(plan)
    }

    pub async fn scan_parquet_row_group_batches(
        &self,
        path: PathBuf,
        batch_size: usize,
        projection: Projection,
        row_groups: Vec<usize>,
    ) -> Result<Vec<RecordBatch>> {
        self.scan_parquet_row_group_batches_profiled(path, batch_size, projection, row_groups)
            .await
            .map(|(batches, _)| batches)
    }

    pub(crate) async fn scan_parquet_row_group_batches_profiled(
        &self,
        path: PathBuf,
        batch_size: usize,
        projection: Projection,
        row_groups: Vec<usize>,
    ) -> Result<(Vec<RecordBatch>, RowGroupBatchScanProfile)> {
        self.scan_parquet_row_group_batches_profiled_inner(
            path,
            batch_size,
            projection,
            row_groups,
            Vec::new(),
        )
        .await
    }

    pub(crate) async fn scan_parquet_row_group_batches_filtered_profiled(
        &self,
        path: PathBuf,
        batch_size: usize,
        projection: Projection,
        row_groups: Vec<usize>,
        row_filter_predicates: Vec<Expr>,
    ) -> Result<(Vec<RecordBatch>, RowGroupBatchScanProfile)> {
        self.scan_parquet_row_group_batches_profiled_inner(
            path,
            batch_size,
            projection,
            row_groups,
            row_filter_predicates,
        )
        .await
    }

    async fn scan_parquet_row_group_batches_profiled_inner(
        &self,
        path: PathBuf,
        batch_size: usize,
        projection: Projection,
        row_groups: Vec<usize>,
        row_filter_predicates: Vec<Expr>,
    ) -> Result<(Vec<RecordBatch>, RowGroupBatchScanProfile)> {
        if row_groups.is_empty() {
            return Ok((Vec::new(), RowGroupBatchScanProfile::default()));
        }
        let source = self.plan_table_source(path).await?;
        if source.format != StorageFormat::Parquet || source.fragments.len() != 1 {
            return Err(DodamError::UnsupportedSql(
                "row-group scan requires a single parquet source".to_string(),
            ));
        }
        let local_path = source.fragments[0].parquet_local_path()?;
        let mut reader = if row_filter_predicates.is_empty() {
            ParquetBatchReader::try_new_with_row_groups(
                local_path,
                batch_size,
                &projection,
                row_groups,
                &self.metadata_cache,
                self.file_cache.clone(),
                self.object_store.as_ref(),
            )?
        } else {
            ParquetBatchReader::try_new_with_row_groups_filtered(
                local_path,
                batch_size,
                &projection,
                row_groups,
                &row_filter_predicates,
                &self.metadata_cache,
                self.file_cache.clone(),
                self.object_store.as_ref(),
            )?
        };
        let mut batches = Vec::new();
        while let Some(batch) = reader.next() {
            let batch = batch?;
            if batch.num_rows() > 0 {
                batches.push(batch);
            }
        }
        let profile = RowGroupBatchScanProfile {
            projected_columns: reader.projected_columns(),
            row_groups_total: reader.row_groups_total(),
            row_groups_scanned: reader.row_groups_scanned(),
            compressed_bytes_total: reader.compressed_bytes_total(),
            compressed_bytes_scanned: reader.compressed_bytes_scanned(),
            metadata_nanos: reader.metadata_nanos(),
            planning_nanos: reader.planning_nanos(),
            next_nanos: reader.next_nanos(),
            max_next_nanos: reader.max_next_nanos(),
            p95_next_nanos: reader.p95_next_nanos(),
            next_calls: reader.next_calls(),
            eof_calls: reader.eof_calls(),
            output_batches: reader.output_batches(),
            output_rows: reader.output_rows(),
            zero_row_batches: reader.zero_row_batches(),
        };
        Ok((batches, profile))
    }

    pub async fn parquet_i64_column_max(&self, path: PathBuf, column: &str) -> Result<Option<i64>> {
        let source = self.plan_table_source(path).await?;
        if source.format != StorageFormat::Parquet || source.fragments.len() != 1 {
            return Ok(None);
        }
        read_parquet_i64_column_max(
            source.fragments[0].parquet_local_path()?,
            column,
            &self.metadata_cache,
            self.object_store.as_ref(),
        )
    }

    pub async fn parquet_i64_column_constant(
        &self,
        path: PathBuf,
        column: &str,
    ) -> Result<Option<i64>> {
        let source = self.plan_table_source(path).await?;
        if source.format != StorageFormat::Parquet || source.fragments.len() != 1 {
            return Ok(None);
        }
        read_parquet_i64_column_constant(
            source.fragments[0].parquet_local_path()?,
            column,
            &self.metadata_cache,
            self.object_store.as_ref(),
        )
    }

    pub async fn parquet_row_groups_monotonic_by_column(
        &self,
        path: PathBuf,
        column: &str,
    ) -> Result<bool> {
        let source = self.plan_table_source(path).await?;
        if source.format != StorageFormat::Parquet || source.fragments.len() != 1 {
            return Ok(false);
        }
        parquet_row_groups_monotonic_by_column(
            source.fragments[0].parquet_local_path()?,
            column,
            &self.metadata_cache,
            self.object_store.as_ref(),
        )
    }

    pub async fn parquet_column_monotonic_by_scan(
        &self,
        path: PathBuf,
        column: &str,
        batch_size: usize,
    ) -> Result<bool> {
        let source = self.plan_table_source(path).await?;
        if source.format != StorageFormat::Parquet || source.fragments.len() != 1 {
            return Ok(false);
        }
        let path = source.fragments[0].parquet_local_path()?;
        let object_metadata = self.object_store.metadata(&path)?;
        let key = MonotonicColumnScanCacheKey {
            path: path.to_path_buf(),
            len: object_metadata.len,
            modified_nanos: object_metadata
                .modified
                .and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
                .map(|duration| duration.as_nanos()),
            column: column.to_string(),
        };
        if let Some(value) = self
            .monotonic_column_scan_cache
            .lock()
            .expect("monotonic column scan cache lock")
            .get(&key)
            .copied()
        {
            return Ok(value);
        }
        let value = parquet_column_monotonic_by_scan(
            &path,
            column,
            batch_size,
            &self.metadata_cache,
            self.object_store.as_ref(),
        )?;
        self.monotonic_column_scan_cache
            .lock()
            .expect("monotonic column scan cache lock")
            .insert(key, value);
        Ok(value)
    }

    pub async fn ordered_i64_decimal_group_sum_above(
        &self,
        path: PathBuf,
        batch_size: usize,
        key_column: &str,
        value_column: &str,
        threshold: f64,
    ) -> Result<Option<HashMap<i64, f64>>> {
        let source = self.plan_table_source(path.clone()).await?;
        if source.format != StorageFormat::Parquet || source.fragments.len() != 1 {
            return Ok(None);
        }
        let projection =
            Projection::Columns(vec![key_column.to_string(), value_column.to_string()]);
        let plan = plan_parquet_scan_tasks(
            source.fragments[0].parquet_local_path()?,
            &projection,
            &[],
            &self.metadata_cache,
            self.object_store.as_ref(),
        )?;
        let row_groups = plan
            .tasks
            .into_iter()
            .map(|task| task.row_group)
            .collect::<Vec<_>>();
        if row_groups.is_empty() {
            return Ok(Some(HashMap::new()));
        }
        let row_group_chunk = ordered_group_sum_row_group_chunk();
        let chunks = row_groups
            .chunks(row_group_chunk)
            .map(|chunk| chunk.to_vec())
            .collect::<Vec<_>>();
        let (sender, receiver) = mpsc::channel();
        for (index, row_groups) in chunks.iter().cloned().enumerate() {
            let sender = sender.clone();
            let path = path.clone();
            let projection = projection.clone();
            let key_column = key_column.to_string();
            let value_column = value_column.to_string();
            let metadata_cache = self.metadata_cache.clone();
            let file_cache = self.file_cache.clone();
            let object_store = self.object_store.clone();
            rayon::spawn(move || {
                let result = ordered_i64_decimal_group_sum_chunk(
                    path,
                    batch_size,
                    &projection,
                    row_groups,
                    &key_column,
                    &value_column,
                    threshold,
                    &metadata_cache,
                    file_cache,
                    object_store.as_ref(),
                );
                let _ = sender.send((index, result));
            });
        }
        drop(sender);

        let mut partials = (0..chunks.len())
            .map(|_| None)
            .collect::<Vec<Option<OrderedGroupSumPartial>>>();
        for _ in 0..chunks.len() {
            let (index, result) = receiver.recv().map_err(|_| {
                DodamError::UnsupportedSql("ordered group worker stopped".to_string())
            })?;
            partials[index] = result?;
        }
        merge_ordered_group_sum_partials(partials, threshold)
    }

    pub async fn scan_parquet_batches_i64_set_filtered(
        &self,
        path: PathBuf,
        batch_size: usize,
        projection: Projection,
        filter_column: &str,
        keys: HashSet<i64>,
    ) -> Result<SendableBatchStream> {
        self.scan_parquet_batches_i64_set_filtered_with_row_group_chunk(
            path,
            batch_size,
            projection,
            filter_column,
            keys,
            parallel_i64_set_filter_row_group_chunk(),
        )
        .await
    }

    pub(crate) async fn scan_parquet_batches_i64_set_filtered_with_row_group_chunk(
        &self,
        path: PathBuf,
        batch_size: usize,
        projection: Projection,
        filter_column: &str,
        keys: HashSet<i64>,
        row_group_chunk: usize,
    ) -> Result<SendableBatchStream> {
        let source = self.plan_table_source(path.clone()).await?;
        if source.format != StorageFormat::Parquet || source.fragments.len() != 1 {
            return self
                .scan_parquet_batches(path, batch_size, None, projection, None)
                .await;
        }
        let plan = plan_parquet_scan_tasks(
            source.fragments[0].parquet_local_path()?,
            &projection,
            &[],
            &self.metadata_cache,
            self.object_store.as_ref(),
        )?;
        let row_groups = plan
            .tasks
            .into_iter()
            .map(|task| task.row_group)
            .collect::<Vec<_>>();
        if row_groups.is_empty() {
            return Ok(SendableBatchStream::empty());
        }
        let chunks = row_groups
            .chunks(row_group_chunk.max(1))
            .map(|chunk| chunk.to_vec())
            .collect::<Vec<_>>();
        let keys = Arc::new(keys);
        let (sender, receiver) = mpsc::channel();
        for row_groups in chunks {
            let sender = sender.clone();
            let path = path.clone();
            let projection = projection.clone();
            let filter_column = filter_column.to_string();
            let keys = keys.clone();
            let metadata_cache = self.metadata_cache.clone();
            let file_cache = self.file_cache.clone();
            let object_store = self.object_store.clone();
            rayon::spawn(move || {
                let result = scan_i64_set_filtered_row_groups(
                    path,
                    batch_size,
                    &projection,
                    row_groups,
                    &filter_column,
                    keys,
                    &metadata_cache,
                    file_cache,
                    object_store.as_ref(),
                );
                match result {
                    Ok(batches) => {
                        for batch in batches {
                            if sender.send(Ok(batch)).is_err() {
                                return;
                            }
                        }
                    }
                    Err(error) => {
                        let _ = sender.send(Err(error));
                    }
                }
            });
        }
        drop(sender);
        Ok(SendableBatchStream::new(
            Box::new(receiver.into_iter()),
            Arc::default(),
        ))
    }

    pub async fn scan_parquet_batches_i64_bloom_filtered(
        &self,
        path: PathBuf,
        batch_size: usize,
        projection: Projection,
        filter_column: &str,
        keys: HashSet<i64>,
    ) -> Result<SendableBatchStream> {
        let source = self.plan_table_source(path.clone()).await?;
        if source.format != StorageFormat::Parquet || source.fragments.len() != 1 {
            return self
                .scan_parquet_batches(path, batch_size, None, projection, None)
                .await;
        }
        let plan = plan_parquet_scan_tasks(
            source.fragments[0].parquet_local_path()?,
            &projection,
            &[],
            &self.metadata_cache,
            self.object_store.as_ref(),
        )?;
        let row_groups = plan
            .tasks
            .into_iter()
            .map(|task| task.row_group)
            .collect::<Vec<_>>();
        if row_groups.is_empty() {
            return Ok(SendableBatchStream::empty());
        }
        let chunks = row_groups
            .chunks(parallel_i64_set_filter_row_group_chunk())
            .map(|chunk| chunk.to_vec())
            .collect::<Vec<_>>();
        let bloom = Arc::new(I64BloomPredicate::from_hash_set(&keys));
        let (sender, receiver) = mpsc::channel();
        for row_groups in chunks {
            let sender = sender.clone();
            let path = path.clone();
            let projection = projection.clone();
            let filter_column = filter_column.to_string();
            let bloom = bloom.clone();
            let metadata_cache = self.metadata_cache.clone();
            let file_cache = self.file_cache.clone();
            let object_store = self.object_store.clone();
            rayon::spawn(move || {
                let result = scan_i64_bloom_filtered_row_groups(
                    path,
                    batch_size,
                    &projection,
                    row_groups,
                    &filter_column,
                    bloom,
                    &metadata_cache,
                    file_cache,
                    object_store.as_ref(),
                );
                match result {
                    Ok(batches) => {
                        for batch in batches {
                            if sender.send(Ok(batch)).is_err() {
                                return;
                            }
                        }
                    }
                    Err(error) => {
                        let _ = sender.send(Err(error));
                    }
                }
            });
        }
        drop(sender);
        Ok(SendableBatchStream::new(
            Box::new(receiver.into_iter()),
            Arc::default(),
        ))
    }

    pub async fn scan_parquet_batches_dictionary_columns(
        &self,
        path: PathBuf,
        batch_size: usize,
        projection: Projection,
        dictionary_columns: Vec<String>,
    ) -> Result<SendableBatchStream> {
        let source = self.plan_table_source(path.clone()).await?;
        if source.format != StorageFormat::Parquet
            || source.fragments.len() != 1
            || dictionary_columns.is_empty()
        {
            return self
                .scan_parquet_batches(path, batch_size, None, projection, None)
                .await;
        }
        let plan = plan_parquet_scan_tasks(
            source.fragments[0].parquet_local_path()?,
            &projection,
            &[],
            &self.metadata_cache,
            self.object_store.as_ref(),
        )?;
        let row_groups = plan
            .tasks
            .into_iter()
            .map(|task| task.row_group)
            .collect::<Vec<_>>();
        if row_groups.is_empty() {
            return Ok(SendableBatchStream::empty());
        }
        let chunks = row_groups
            .chunks(parallel_i64_set_filter_row_group_chunk())
            .map(|chunk| chunk.to_vec())
            .collect::<Vec<_>>();
        let dictionary_columns = Arc::new(dictionary_columns);
        let (sender, receiver) = mpsc::channel();
        for row_groups in chunks {
            let sender = sender.clone();
            let path = path.clone();
            let projection = projection.clone();
            let dictionary_columns = dictionary_columns.clone();
            let metadata_cache = self.metadata_cache.clone();
            let file_cache = self.file_cache.clone();
            let object_store = self.object_store.clone();
            rayon::spawn(move || {
                let result = scan_dictionary_column_row_groups(
                    path,
                    batch_size,
                    &projection,
                    row_groups,
                    dictionary_columns.as_ref(),
                    &metadata_cache,
                    file_cache,
                    object_store.as_ref(),
                );
                match result {
                    Ok(batches) => {
                        for batch in batches {
                            if sender.send(Ok(batch)).is_err() {
                                return;
                            }
                        }
                    }
                    Err(error) => {
                        let _ = sender.send(Err(error));
                    }
                }
            });
        }
        drop(sender);
        Ok(SendableBatchStream::new(
            Box::new(receiver.into_iter()),
            Arc::default(),
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn q06_late_materialized_revenue_sum(
        &self,
        path: PathBuf,
        batch_size: usize,
        start_days: i32,
        end_days: i32,
        discount_low: f64,
        discount_high: f64,
        quantity_limit: f64,
    ) -> Result<Option<(f64, u64)>> {
        let predicate_projection = Projection::Columns(vec![
            "l_shipdate".to_string(),
            "l_discount".to_string(),
            "l_quantity".to_string(),
        ]);
        let payload_projection = Projection::Columns(vec!["l_extendedprice".to_string()]);
        let chunks = self
            .late_materialized_parquet_map_pruned_with_policy_view(
                path,
                batch_size,
                predicate_projection,
                payload_projection,
                Vec::new(),
                selected_discount_revenue_row_group_chunk(),
                LateMaterializationPolicy::always(),
                || SelectedDiscountRevenueState {
                    selected_discounts: Vec::new(),
                    discount_scale: None,
                    extendedprice_scale: None,
                    discount_offset: 0,
                    revenue: 0.0,
                },
                move |view, selection, state| {
                    build_date32_discount_quantity_selection_view(
                        view,
                        start_days,
                        end_days,
                        discount_low,
                        discount_high,
                        quantity_limit,
                        selection,
                        state,
                    )
                },
                consume_selected_discount_revenue_view,
                |state, _metrics| {
                    if state.discount_offset != state.selected_discounts.len() {
                        return Err(DodamError::UnsupportedSql(
                            "selected discount revenue payload mismatch".to_string(),
                        ));
                    }
                    Ok(Some((state.revenue, state.selected_discounts.len() as u64)))
                },
            )
            .await?;
        let Some(chunks) = chunks else {
            return Ok(None);
        };
        let chunk_count = chunks.len();
        let mut sum = 0.0;
        let mut count = 0_u64;
        let mut metrics = LateMaterializedMetrics::default();
        for chunk in chunks {
            let (partial_sum, partial_count) = chunk.output;
            sum += partial_sum;
            count += partial_count;
            metrics.add(chunk.metrics);
        }
        log_late_materialized_metrics("Q06", metrics, chunk_count);
        Ok(Some((sum, count)))
    }

    pub(crate) async fn q14_late_materialized_promo_revenue(
        &self,
        path: PathBuf,
        batch_size: usize,
        start_days: i32,
        end_days: i32,
        lookup: Arc<DenseI64BoolLookup>,
    ) -> Result<Option<(f64, f64)>> {
        let predicate_projection = Projection::Columns(vec!["l_shipdate".to_string()]);
        let payload_projection = Projection::Columns(vec![
            "l_partkey".to_string(),
            "l_extendedprice".to_string(),
            "l_discount".to_string(),
        ]);
        let chunks = self
            .late_materialized_parquet_map_pruned_with_policy_view(
                path,
                batch_size,
                predicate_projection,
                payload_projection,
                Vec::new(),
                bool_lookup_discounted_revenue_row_group_chunk(),
                LateMaterializationPolicy::always(),
                {
                    let lookup = lookup.clone();
                    move || BoolLookupDiscountedRevenueState {
                        lookup: lookup.clone(),
                        matched: 0.0,
                        total: 0.0,
                    }
                },
                move |view, selection, _state| {
                    build_date32_range_selection_view(view, start_days, end_days, selection)
                },
                consume_discounted_revenue_by_i64_bool_lookup_view,
                |state, _metrics| Ok(Some((state.matched, state.total))),
            )
            .await?;
        let Some(chunks) = chunks else {
            return Ok(None);
        };
        let chunk_count = chunks.len();
        let mut promo = 0.0;
        let mut total = 0.0;
        let mut metrics = LateMaterializedMetrics::default();
        for chunk in chunks {
            let (partial_promo, partial_total) = chunk.output;
            promo += partial_promo;
            total += partial_total;
            metrics.add(chunk.metrics);
        }
        log_late_materialized_metrics("Q14", metrics, chunk_count);
        Ok(Some((promo, total)))
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn late_materialized_parquet_map<
        State,
        Output,
        BuildState,
        BuildSelection,
        ConsumePayload,
        Finish,
    >(
        &self,
        path: PathBuf,
        batch_size: usize,
        predicate_projection: Projection,
        payload_projection: Projection,
        row_group_chunk: usize,
        build_state: BuildState,
        build_selection: BuildSelection,
        consume_payload: ConsumePayload,
        finish: Finish,
    ) -> Result<Option<Vec<LateMaterializedChunkResult<Output>>>>
    where
        State: Send + 'static,
        Output: Send + 'static,
        BuildState: Fn() -> State + Clone + Send + Sync + 'static,
        BuildSelection: Fn(RecordBatch, &mut LateSelectionBuilder, &mut State) -> Result<Option<()>>
            + Clone
            + Send
            + Sync
            + 'static,
        ConsumePayload:
            Fn(RecordBatch, &mut State) -> Result<Option<()>> + Clone + Send + Sync + 'static,
        Finish: Fn(State, LateMaterializedMetrics) -> Result<Option<Output>>
            + Clone
            + Send
            + Sync
            + 'static,
    {
        self.late_materialized_parquet_map_pruned_with_policy(
            path,
            batch_size,
            predicate_projection,
            payload_projection,
            Vec::new(),
            row_group_chunk,
            LateMaterializationPolicy::always(),
            build_state,
            build_selection,
            consume_payload,
            finish,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn late_materialized_parquet_map_with_policy<
        State,
        Output,
        BuildState,
        BuildSelection,
        ConsumePayload,
        Finish,
    >(
        &self,
        path: PathBuf,
        batch_size: usize,
        predicate_projection: Projection,
        payload_projection: Projection,
        row_group_chunk: usize,
        policy: LateMaterializationPolicy,
        build_state: BuildState,
        build_selection: BuildSelection,
        consume_payload: ConsumePayload,
        finish: Finish,
    ) -> Result<Option<Vec<LateMaterializedChunkResult<Output>>>>
    where
        State: Send + 'static,
        Output: Send + 'static,
        BuildState: Fn() -> State + Clone + Send + Sync + 'static,
        BuildSelection: Fn(RecordBatch, &mut LateSelectionBuilder, &mut State) -> Result<Option<()>>
            + Clone
            + Send
            + Sync
            + 'static,
        ConsumePayload:
            Fn(RecordBatch, &mut State) -> Result<Option<()>> + Clone + Send + Sync + 'static,
        Finish: Fn(State, LateMaterializedMetrics) -> Result<Option<Output>>
            + Clone
            + Send
            + Sync
            + 'static,
    {
        self.late_materialized_parquet_map_pruned_with_policy(
            path,
            batch_size,
            predicate_projection,
            payload_projection,
            Vec::new(),
            row_group_chunk,
            policy,
            build_state,
            build_selection,
            consume_payload,
            finish,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn late_materialized_parquet_map_pruned_with_policy<
        State,
        Output,
        BuildState,
        BuildSelection,
        ConsumePayload,
        Finish,
    >(
        &self,
        path: PathBuf,
        batch_size: usize,
        predicate_projection: Projection,
        payload_projection: Projection,
        pruning_predicates: Vec<Expr>,
        row_group_chunk: usize,
        policy: LateMaterializationPolicy,
        build_state: BuildState,
        build_selection: BuildSelection,
        consume_payload: ConsumePayload,
        finish: Finish,
    ) -> Result<Option<Vec<LateMaterializedChunkResult<Output>>>>
    where
        State: Send + 'static,
        Output: Send + 'static,
        BuildState: Fn() -> State + Clone + Send + Sync + 'static,
        BuildSelection: Fn(RecordBatch, &mut LateSelectionBuilder, &mut State) -> Result<Option<()>>
            + Clone
            + Send
            + Sync
            + 'static,
        ConsumePayload:
            Fn(RecordBatch, &mut State) -> Result<Option<()>> + Clone + Send + Sync + 'static,
        Finish: Fn(State, LateMaterializedMetrics) -> Result<Option<Output>>
            + Clone
            + Send
            + Sync
            + 'static,
    {
        self.late_materialized_parquet_map_pruned_with_policy_view(
            path,
            batch_size,
            predicate_projection,
            payload_projection,
            pruning_predicates,
            row_group_chunk,
            policy,
            build_state,
            move |view, selection, state| {
                let Some(batch) = view.try_record_batch() else {
                    return Err(DodamError::UnsupportedSql(
                        "late materialized selection raw vector fallback requires RecordBatch"
                            .to_string(),
                    ));
                };
                build_selection(batch.clone(), selection, state)
            },
            move |view, state| {
                let Some(batch) = view.try_record_batch() else {
                    return Err(DodamError::UnsupportedSql(
                        "late materialized payload raw vector fallback requires RecordBatch"
                            .to_string(),
                    ));
                };
                consume_payload(batch.clone(), state)
            },
            finish,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn late_materialized_parquet_map_pruned_with_policy_view<
        State,
        Output,
        BuildState,
        BuildSelection,
        ConsumePayload,
        Finish,
    >(
        &self,
        path: PathBuf,
        batch_size: usize,
        predicate_projection: Projection,
        payload_projection: Projection,
        pruning_predicates: Vec<Expr>,
        row_group_chunk: usize,
        policy: LateMaterializationPolicy,
        build_state: BuildState,
        build_selection: BuildSelection,
        consume_payload: ConsumePayload,
        finish: Finish,
    ) -> Result<Option<Vec<LateMaterializedChunkResult<Output>>>>
    where
        State: Send + 'static,
        Output: Send + 'static,
        BuildState: Fn() -> State + Clone + Send + Sync + 'static,
        BuildSelection: for<'a> FnMut(
                BatchView<'a>,
                &mut LateSelectionBuilder,
                &mut State,
            ) -> Result<Option<()>>
            + Clone
            + Send
            + Sync
            + 'static,
        ConsumePayload: for<'a> FnMut(BatchView<'a>, &mut State) -> Result<Option<()>>
            + Clone
            + Send
            + Sync
            + 'static,
        Finish: Fn(State, LateMaterializedMetrics) -> Result<Option<Output>>
            + Clone
            + Send
            + Sync
            + 'static,
    {
        self.late_materialized_parquet_map_pruned_with_policy_view_dictionary_columns(
            path,
            batch_size,
            predicate_projection,
            payload_projection,
            Vec::new(),
            pruning_predicates,
            row_group_chunk,
            policy,
            build_state,
            build_selection,
            consume_payload,
            finish,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn late_materialized_parquet_map_pruned_with_policy_view_dictionary_columns<
        State,
        Output,
        BuildState,
        BuildSelection,
        ConsumePayload,
        Finish,
    >(
        &self,
        path: PathBuf,
        batch_size: usize,
        predicate_projection: Projection,
        payload_projection: Projection,
        payload_dictionary_columns: Vec<String>,
        pruning_predicates: Vec<Expr>,
        row_group_chunk: usize,
        policy: LateMaterializationPolicy,
        build_state: BuildState,
        build_selection: BuildSelection,
        consume_payload: ConsumePayload,
        finish: Finish,
    ) -> Result<Option<Vec<LateMaterializedChunkResult<Output>>>>
    where
        State: Send + 'static,
        Output: Send + 'static,
        BuildState: Fn() -> State + Clone + Send + Sync + 'static,
        BuildSelection: for<'a> FnMut(
                BatchView<'a>,
                &mut LateSelectionBuilder,
                &mut State,
            ) -> Result<Option<()>>
            + Clone
            + Send
            + Sync
            + 'static,
        ConsumePayload: for<'a> FnMut(BatchView<'a>, &mut State) -> Result<Option<()>>
            + Clone
            + Send
            + Sync
            + 'static,
        Finish: Fn(State, LateMaterializedMetrics) -> Result<Option<Output>>
            + Clone
            + Send
            + Sync
            + 'static,
    {
        let source = self.plan_table_source(path.clone()).await?;
        if source.format != StorageFormat::Parquet || source.fragments.len() != 1 {
            return Ok(None);
        }
        let local_path = source.fragments[0].parquet_local_path()?.to_path_buf();
        let plan = plan_parquet_scan_tasks(
            &local_path,
            &predicate_projection,
            &pruning_predicates,
            &self.metadata_cache,
            self.object_store.as_ref(),
        )?;
        let row_groups = plan
            .tasks
            .into_iter()
            .map(|task| task.row_group)
            .collect::<Vec<_>>();
        if row_groups.is_empty() {
            return Ok(Some(Vec::new()));
        }
        let (predicate_compressed_bytes, payload_compressed_bytes) = if policy.io_cost_gate {
            (
                plan.compressed_bytes_scanned,
                plan_parquet_scan_tasks(
                    &local_path,
                    &payload_projection,
                    &pruning_predicates,
                    &self.metadata_cache,
                    self.object_store.as_ref(),
                )?
                .compressed_bytes_scanned,
            )
        } else {
            (0, 0)
        };
        let payload_columns = match &payload_projection {
            Projection::All => source
                .schema
                .as_ref()
                .map_or(plan.schema_columns, |schema| schema.fields().len()),
            Projection::Columns(columns) => columns.len(),
        };
        if late_materialization_sample_enabled(policy, plan.projected_columns, payload_columns) {
            let sample_row_groups = row_groups
                .iter()
                .copied()
                .take(late_materialization_sample_row_groups())
                .collect::<Vec<_>>();
            let Some(sample_metrics) = sample_late_materialized_selection_view(
                local_path.clone(),
                batch_size,
                sample_row_groups,
                &predicate_projection,
                &self.metadata_cache,
                self.file_cache.clone(),
                self.object_store.as_ref(),
                build_state.clone()(),
                build_selection.clone(),
            )?
            else {
                return Ok(None);
            };
            let accepted = late_materialization_policy_accepts_with_io(
                policy,
                &sample_metrics,
                predicate_compressed_bytes,
                payload_compressed_bytes,
            );
            log_late_materialization_policy_decision(
                "sample",
                accepted,
                &sample_metrics,
                predicate_compressed_bytes,
                payload_compressed_bytes,
                policy,
            );
            if !accepted {
                return Ok(None);
            }
        }
        let chunks = row_groups
            .chunks(row_group_chunk.max(1))
            .map(|chunk| chunk.to_vec())
            .collect::<Vec<_>>();
        let (sender, receiver) = mpsc::channel();
        for (index, row_groups) in chunks.iter().cloned().enumerate() {
            let sender = sender.clone();
            let path = local_path.clone();
            let predicate_projection = predicate_projection.clone();
            let payload_projection = payload_projection.clone();
            let payload_dictionary_columns = payload_dictionary_columns.clone();
            let metadata_cache = self.metadata_cache.clone();
            let file_cache = self.file_cache.clone();
            let object_store = self.object_store.clone();
            let build_state = build_state.clone();
            let build_selection = build_selection.clone();
            let consume_payload = consume_payload.clone();
            let finish = finish.clone();
            rayon::spawn(move || {
                let result = late_materialized_chunk_view(
                    path,
                    batch_size,
                    row_groups,
                    &predicate_projection,
                    &payload_projection,
                    &payload_dictionary_columns,
                    &metadata_cache,
                    file_cache,
                    object_store.as_ref(),
                    policy,
                    predicate_compressed_bytes,
                    payload_compressed_bytes,
                    build_state(),
                    build_selection,
                    consume_payload,
                    finish,
                );
                let _ = sender.send((index, result));
            });
        }
        drop(sender);

        let mut outputs = (0..chunks.len())
            .map(|_| None)
            .collect::<Vec<Option<Option<LateMaterializedChunkResult<Output>>>>>();
        for _ in 0..chunks.len() {
            let (index, result) = receiver.recv().map_err(|_| {
                DodamError::UnsupportedSql("late materialized view worker stopped".to_string())
            })?;
            outputs[index] = Some(result?);
        }
        let mut results = Vec::with_capacity(outputs.len());
        for output in outputs {
            let Some(output) = output else {
                return Err(DodamError::UnsupportedSql(
                    "late materialized view worker result missing".to_string(),
                ));
            };
            let Some(output) = output else {
                return Ok(None);
            };
            results.push(output);
        }
        Ok(Some(results))
    }

    pub async fn parquet_row_group_map<State, Output, BuildState, ConsumeBatch, Finish>(
        &self,
        path: PathBuf,
        batch_size: usize,
        projection: Projection,
        row_group_chunk: usize,
        build_state: BuildState,
        consume_batch: ConsumeBatch,
        finish: Finish,
    ) -> Result<Option<Vec<Output>>>
    where
        State: Send + 'static,
        Output: Send + 'static,
        BuildState: Fn() -> State + Clone + Send + Sync + 'static,
        ConsumeBatch:
            FnMut(RecordBatch, &mut State) -> Result<Option<()>> + Clone + Send + Sync + 'static,
        Finish: Fn(State) -> Result<Option<Output>> + Clone + Send + Sync + 'static,
    {
        Ok(self
            .parquet_row_group_map_results(
                "row_group_map",
                path,
                batch_size,
                projection,
                Vec::new(),
                row_group_chunk,
                build_state,
                consume_batch,
                finish,
            )
            .await?
            .map(|results| results.outputs))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn parquet_row_group_map_view<
        State,
        Output,
        BuildState,
        ConsumeBatch,
        Finish,
    >(
        &self,
        path: PathBuf,
        batch_size: usize,
        projection: Projection,
        row_group_chunk: usize,
        build_state: BuildState,
        consume_batch: ConsumeBatch,
        finish: Finish,
    ) -> Result<Option<Vec<Output>>>
    where
        State: Send + 'static,
        Output: Send + 'static,
        BuildState: Fn() -> State + Clone + Send + Sync + 'static,
        ConsumeBatch: for<'a> FnMut(BatchView<'a>, &mut State) -> Result<Option<()>>
            + Clone
            + Send
            + Sync
            + 'static,
        Finish: Fn(State) -> Result<Option<Output>> + Clone + Send + Sync + 'static,
    {
        self.parquet_row_group_map_pruned_view(
            path,
            batch_size,
            projection,
            Vec::new(),
            row_group_chunk,
            build_state,
            consume_batch,
            finish,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn parquet_row_group_map_results<State, Output, BuildState, ConsumeBatch, Finish>(
        &self,
        profile_kind: &str,
        path: PathBuf,
        batch_size: usize,
        projection: Projection,
        pruning_predicates: Vec<Expr>,
        row_group_chunk: usize,
        build_state: BuildState,
        consume_batch: ConsumeBatch,
        finish: Finish,
    ) -> Result<Option<ParquetMapResults<Output>>>
    where
        State: Send + 'static,
        Output: Send + 'static,
        BuildState: Fn() -> State + Clone + Send + Sync + 'static,
        ConsumeBatch:
            FnMut(RecordBatch, &mut State) -> Result<Option<()>> + Clone + Send + Sync + 'static,
        Finish: Fn(State) -> Result<Option<Output>> + Clone + Send + Sync + 'static,
    {
        self.parquet_row_group_map_pruned_results(
            profile_kind,
            path,
            batch_size,
            projection,
            pruning_predicates,
            row_group_chunk,
            build_state,
            consume_batch,
            finish,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn parquet_scan_fold_chunks<Output, BuildPartial, BuildOutput, Map, Merge>(
        &self,
        path: PathBuf,
        batch_size: usize,
        projection: Projection,
        row_group_chunk: usize,
        stream_chunk: usize,
        enable_row_group_map: bool,
        build_partial: BuildPartial,
        build_output: BuildOutput,
        map: Map,
        mut merge: Merge,
        label: &str,
    ) -> Result<Output>
    where
        Output: Send + 'static,
        BuildPartial: Fn() -> Output + Clone + Send + Sync + 'static,
        BuildOutput: Fn() -> Output + Clone + Send + Sync + 'static,
        Map: Fn(RecordBatch) -> Result<Output> + Clone + Send + Sync + 'static,
        Merge: FnMut(&mut Output, Output) + Clone + Send + Sync + 'static,
    {
        if enable_row_group_map
            && let Some(partials) = self
                .parquet_row_group_map_results(
                    "row_group_map",
                    path.clone(),
                    batch_size,
                    projection.clone(),
                    Vec::new(),
                    row_group_chunk,
                    build_partial.clone(),
                    {
                        let map = map.clone();
                        let merge = merge.clone();
                        move |batch, output| {
                            let mut merge = merge.clone();
                            merge(output, map(batch)?);
                            Ok(Some(()))
                        }
                    },
                    |output| Ok(Some(output)),
                )
                .await?
        {
            let profile = scan_profile_enabled();
            let started = profile.then(Instant::now);
            let mut output = build_output();
            for partial in partials.outputs {
                merge(&mut output, partial);
            }
            if let Some(started) = started {
                eprintln!(
                    "[dodam:scan-fold-profile] {label}: fused_merge={:.3} ms scan_total_sum={:.3} ms scan_read_next={:.3} ms scan_consume={:.3} ms chunks={} row_groups={} batches={} rows={} row_group_chunk={row_group_chunk}",
                    started.elapsed().as_secs_f64() * 1000.0,
                    nanos_to_millis(partials.metrics.total_nanos),
                    nanos_to_millis(partials.metrics.read_nanos),
                    nanos_to_millis(partials.metrics.consume_nanos),
                    partials.metrics.chunks,
                    partials.metrics.row_groups,
                    partials.metrics.batches,
                    partials.metrics.rows,
                );
            }
            return Ok(output);
        }

        let mut stream = self
            .scan_parquet_batches(path, batch_size, None, projection, None)
            .await?;
        self.fold_scan_stream_chunks(
            &mut stream,
            stream_chunk,
            build_partial,
            build_output,
            map,
            merge,
            label,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn parquet_scan_accumulate_chunks<Output, BuildPartial, BuildOutput, Consume, Merge>(
        &self,
        path: PathBuf,
        batch_size: usize,
        projection: Projection,
        row_group_chunk: usize,
        stream_chunk: usize,
        enable_row_group_map: bool,
        build_partial: BuildPartial,
        build_output: BuildOutput,
        consume_batch: Consume,
        mut merge: Merge,
        label: &str,
    ) -> Result<Output>
    where
        Output: Send + 'static,
        BuildPartial: Fn() -> Output + Clone + Send + Sync + 'static,
        BuildOutput: Fn() -> Output + Clone + Send + Sync + 'static,
        Consume:
            FnMut(RecordBatch, &mut Output) -> Result<Option<()>> + Clone + Send + Sync + 'static,
        Merge: FnMut(&mut Output, Output) + Clone + Send + Sync + 'static,
    {
        if enable_row_group_map
            && let Some(partials) = self
                .parquet_row_group_map_results(
                    "row_group_map",
                    path.clone(),
                    batch_size,
                    projection.clone(),
                    Vec::new(),
                    row_group_chunk,
                    build_partial.clone(),
                    consume_batch.clone(),
                    |output| Ok(Some(output)),
                )
                .await?
        {
            let profile = scan_profile_enabled();
            let started = profile.then(Instant::now);
            let mut output = build_output();
            for partial in partials.outputs {
                merge(&mut output, partial);
            }
            if let Some(started) = started {
                eprintln!(
                    "[dodam:scan-fold-profile] {label}: accumulate_merge={:.3} ms scan_total_sum={:.3} ms scan_read_next={:.3} ms scan_consume={:.3} ms chunks={} row_groups={} batches={} rows={} row_group_chunk={row_group_chunk}",
                    started.elapsed().as_secs_f64() * 1000.0,
                    nanos_to_millis(partials.metrics.total_nanos),
                    nanos_to_millis(partials.metrics.read_nanos),
                    nanos_to_millis(partials.metrics.consume_nanos),
                    partials.metrics.chunks,
                    partials.metrics.row_groups,
                    partials.metrics.batches,
                    partials.metrics.rows,
                );
            }
            return Ok(output);
        }

        let mut stream = self
            .scan_parquet_batches(path, batch_size, None, projection, None)
            .await?;
        self.accumulate_scan_stream_chunks(
            &mut stream,
            stream_chunk,
            build_partial,
            build_output,
            consume_batch,
            merge,
            label,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn parquet_scan_accumulate_chunks_view<
        Output,
        BuildPartial,
        BuildOutput,
        Consume,
        Merge,
    >(
        &self,
        path: PathBuf,
        batch_size: usize,
        projection: Projection,
        row_group_chunk: usize,
        stream_chunk: usize,
        enable_row_group_map: bool,
        build_partial: BuildPartial,
        build_output: BuildOutput,
        consume_batch: Consume,
        mut merge: Merge,
        label: &str,
    ) -> Result<Output>
    where
        Output: Send + 'static,
        BuildPartial: Fn() -> Output + Clone + Send + Sync + 'static,
        BuildOutput: Fn() -> Output + Clone + Send + Sync + 'static,
        Consume: for<'a> FnMut(BatchView<'a>, &mut Output) -> Result<Option<()>>
            + Clone
            + Send
            + Sync
            + 'static,
        Merge: FnMut(&mut Output, Output) + Clone + Send + Sync + 'static,
    {
        if enable_row_group_map
            && let Some(partials) = self
                .parquet_row_group_map_pruned_view(
                    path.clone(),
                    batch_size,
                    projection.clone(),
                    Vec::new(),
                    row_group_chunk,
                    build_partial.clone(),
                    consume_batch.clone(),
                    |output| Ok(Some(output)),
                )
                .await?
        {
            let profile = scan_profile_enabled();
            let started = profile.then(Instant::now);
            let mut output = build_output();
            for partial in partials {
                merge(&mut output, partial);
            }
            if let Some(started) = started {
                eprintln!(
                    "[dodam:scan-fold-profile] {label}: accumulate_view_merge={:.3} ms row_group_chunk={row_group_chunk}",
                    started.elapsed().as_secs_f64() * 1000.0,
                );
            }
            return Ok(output);
        }

        let mut stream = self
            .scan_parquet_batches(path, batch_size, None, projection, None)
            .await?;
        self.accumulate_scan_stream_view_chunks(
            &mut stream,
            stream_chunk,
            build_partial,
            build_output,
            consume_batch,
            merge,
            label,
        )
    }

    fn fold_scan_stream_chunks<Output, BuildPartial, BuildOutput, Map, Merge>(
        &self,
        stream: &mut SendableBatchStream,
        chunk_size: usize,
        build_partial: BuildPartial,
        build_output: BuildOutput,
        map: Map,
        mut merge: Merge,
        label: &str,
    ) -> Result<Output>
    where
        Output: Send + 'static,
        BuildPartial: Fn() -> Output + Clone + Send + Sync + 'static,
        BuildOutput: Fn() -> Output + Clone + Send + Sync + 'static,
        Map: Fn(RecordBatch) -> Result<Output> + Clone + Send + Sync + 'static,
        Merge: FnMut(&mut Output, Output) + Clone + Send + Sync + 'static,
    {
        let profile = scan_profile_enabled();
        let started = profile.then(Instant::now);
        let (sender, receiver) = mpsc::channel();
        let mut pending_chunks = 0_usize;
        let mut chunk = Vec::with_capacity(chunk_size.max(1));
        let stream_started = profile.then(Instant::now);
        while let Some(batch) = stream.next() {
            chunk.push(batch?);
            if chunk.len() < chunk_size.max(1) {
                continue;
            }
            let sender = sender.clone();
            let map = map.clone();
            let build_partial = build_partial.clone();
            let merge = merge.clone();
            let task_chunk = std::mem::replace(&mut chunk, Vec::with_capacity(chunk_size.max(1)));
            pending_chunks += 1;
            rayon::spawn(move || {
                let mut output = build_partial();
                let mut merge = merge.clone();
                let result = task_chunk
                    .into_iter()
                    .try_for_each(|batch| -> Result<()> {
                        merge(&mut output, map(batch)?);
                        Ok(())
                    })
                    .map(|()| output);
                let _ = sender.send(result);
            });
        }
        if !chunk.is_empty() {
            let sender = sender.clone();
            let map = map.clone();
            let build_partial = build_partial.clone();
            let merge = merge.clone();
            pending_chunks += 1;
            rayon::spawn(move || {
                let mut output = build_partial();
                let mut merge = merge.clone();
                let result = chunk
                    .into_iter()
                    .try_for_each(|batch| -> Result<()> {
                        merge(&mut output, map(batch)?);
                        Ok(())
                    })
                    .map(|()| output);
                let _ = sender.send(result);
            });
        }
        let stream_ms = stream_started
            .map(|started| started.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or_default();
        drop(sender);

        let merge_started = profile.then(Instant::now);
        let mut output = build_output();
        for _ in 0..pending_chunks {
            let partial = receiver.recv().map_err(|_| {
                DodamError::UnsupportedSql(format!("{label} scan-fold worker stopped"))
            })??;
            merge(&mut output, partial);
        }
        if let Some(started) = started {
            let merge_ms = merge_started
                .map(|started| started.elapsed().as_secs_f64() * 1000.0)
                .unwrap_or_default();
            eprintln!(
                "[dodam:scan-fold-profile] {label}: total={:.3} ms stream_read={:.3} ms worker_wait_merge={:.3} ms chunks={pending_chunks}",
                started.elapsed().as_secs_f64() * 1000.0,
                stream_ms,
                merge_ms
            );
        }
        Ok(output)
    }

    fn accumulate_scan_stream_view_chunks<Output, BuildPartial, BuildOutput, Consume, Merge>(
        &self,
        stream: &mut SendableBatchStream,
        chunk_size: usize,
        build_partial: BuildPartial,
        build_output: BuildOutput,
        consume_batch: Consume,
        mut merge: Merge,
        label: &str,
    ) -> Result<Output>
    where
        Output: Send + 'static,
        BuildPartial: Fn() -> Output + Clone + Send + Sync + 'static,
        BuildOutput: Fn() -> Output + Clone + Send + Sync + 'static,
        Consume: for<'a> FnMut(BatchView<'a>, &mut Output) -> Result<Option<()>>
            + Clone
            + Send
            + Sync
            + 'static,
        Merge: FnMut(&mut Output, Output) + Clone + Send + Sync + 'static,
    {
        let profile = scan_profile_enabled();
        let started = profile.then(Instant::now);
        let (sender, receiver) = mpsc::channel();
        let mut pending_chunks = 0_usize;
        let chunk_size = chunk_size.max(1);
        let mut chunk = Vec::with_capacity(chunk_size);
        let stream_started = profile.then(Instant::now);
        while let Some(batch) = stream.next() {
            chunk.push(batch?);
            if chunk.len() < chunk_size {
                continue;
            }
            let sender = sender.clone();
            let build_partial = build_partial.clone();
            let mut consume_batch = consume_batch.clone();
            let task_chunk = std::mem::replace(&mut chunk, Vec::with_capacity(chunk_size));
            pending_chunks += 1;
            rayon::spawn(move || {
                let mut output = build_partial();
                let result = task_chunk
                    .iter()
                    .try_for_each(|batch| -> Result<()> {
                        consume_batch(BatchView::new(batch), &mut output)?.ok_or_else(|| {
                            DodamError::UnsupportedSql(
                                "scan accumulate view worker stopped".to_string(),
                            )
                        })?;
                        Ok(())
                    })
                    .map(|()| output);
                let _ = sender.send(result);
            });
        }
        if !chunk.is_empty() {
            let sender = sender.clone();
            let build_partial = build_partial.clone();
            let mut consume_batch = consume_batch.clone();
            pending_chunks += 1;
            rayon::spawn(move || {
                let mut output = build_partial();
                let result = chunk
                    .iter()
                    .try_for_each(|batch| -> Result<()> {
                        consume_batch(BatchView::new(batch), &mut output)?.ok_or_else(|| {
                            DodamError::UnsupportedSql(
                                "scan accumulate view worker stopped".to_string(),
                            )
                        })?;
                        Ok(())
                    })
                    .map(|()| output);
                let _ = sender.send(result);
            });
        }
        let stream_ms = stream_started
            .map(|started| started.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or_default();
        drop(sender);

        let merge_started = profile.then(Instant::now);
        let mut output = build_output();
        for _ in 0..pending_chunks {
            let partial = receiver.recv().map_err(|_| {
                DodamError::UnsupportedSql(format!("{label} scan-accumulate view worker stopped"))
            })??;
            merge(&mut output, partial);
        }
        if let Some(started) = started {
            let merge_ms = merge_started
                .map(|started| started.elapsed().as_secs_f64() * 1000.0)
                .unwrap_or_default();
            eprintln!(
                "[dodam:scan-fold-profile] {label}: accumulate_view_total={:.3} ms stream_read={:.3} ms worker_wait_merge={:.3} ms chunks={pending_chunks}",
                started.elapsed().as_secs_f64() * 1000.0,
                stream_ms,
                merge_ms
            );
        }
        Ok(output)
    }

    fn accumulate_scan_stream_chunks<Output, BuildPartial, BuildOutput, Consume, Merge>(
        &self,
        stream: &mut SendableBatchStream,
        chunk_size: usize,
        build_partial: BuildPartial,
        build_output: BuildOutput,
        consume_batch: Consume,
        mut merge: Merge,
        label: &str,
    ) -> Result<Output>
    where
        Output: Send + 'static,
        BuildPartial: Fn() -> Output + Clone + Send + Sync + 'static,
        BuildOutput: Fn() -> Output + Clone + Send + Sync + 'static,
        Consume:
            FnMut(RecordBatch, &mut Output) -> Result<Option<()>> + Clone + Send + Sync + 'static,
        Merge: FnMut(&mut Output, Output) + Clone + Send + Sync + 'static,
    {
        let profile = scan_profile_enabled();
        let started = profile.then(Instant::now);
        let (sender, receiver) = mpsc::channel();
        let mut pending_chunks = 0_usize;
        let mut chunk = Vec::with_capacity(chunk_size.max(1));
        let stream_started = profile.then(Instant::now);
        while let Some(batch) = stream.next() {
            chunk.push(batch?);
            if chunk.len() < chunk_size.max(1) {
                continue;
            }
            let sender = sender.clone();
            let build_partial = build_partial.clone();
            let mut consume_batch = consume_batch.clone();
            let task_chunk = std::mem::replace(&mut chunk, Vec::with_capacity(chunk_size.max(1)));
            pending_chunks += 1;
            rayon::spawn(move || {
                let mut output = build_partial();
                let result = task_chunk
                    .into_iter()
                    .try_for_each(|batch| -> Result<()> {
                        consume_batch(batch, &mut output)?.ok_or_else(|| {
                            DodamError::UnsupportedSql("scan accumulate worker stopped".to_string())
                        })?;
                        Ok(())
                    })
                    .map(|()| output);
                let _ = sender.send(result);
            });
        }
        if !chunk.is_empty() {
            let sender = sender.clone();
            let build_partial = build_partial.clone();
            let mut consume_batch = consume_batch.clone();
            pending_chunks += 1;
            rayon::spawn(move || {
                let mut output = build_partial();
                let result = chunk
                    .into_iter()
                    .try_for_each(|batch| -> Result<()> {
                        consume_batch(batch, &mut output)?.ok_or_else(|| {
                            DodamError::UnsupportedSql("scan accumulate worker stopped".to_string())
                        })?;
                        Ok(())
                    })
                    .map(|()| output);
                let _ = sender.send(result);
            });
        }
        let stream_ms = stream_started
            .map(|started| started.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or_default();
        drop(sender);

        let merge_started = profile.then(Instant::now);
        let mut output = build_output();
        for _ in 0..pending_chunks {
            let partial = receiver.recv().map_err(|_| {
                DodamError::UnsupportedSql(format!("{label} scan-accumulate worker stopped"))
            })??;
            merge(&mut output, partial);
        }
        if let Some(started) = started {
            let merge_ms = merge_started
                .map(|started| started.elapsed().as_secs_f64() * 1000.0)
                .unwrap_or_default();
            eprintln!(
                "[dodam:scan-fold-profile] {label}: accumulate_total={:.3} ms stream_read={:.3} ms worker_wait_merge={:.3} ms chunks={pending_chunks}",
                started.elapsed().as_secs_f64() * 1000.0,
                stream_ms,
                merge_ms
            );
        }
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn parquet_row_group_map_pruned<State, Output, BuildState, ConsumeBatch, Finish>(
        &self,
        path: PathBuf,
        batch_size: usize,
        projection: Projection,
        pruning_predicates: Vec<Expr>,
        row_group_chunk: usize,
        build_state: BuildState,
        consume_batch: ConsumeBatch,
        finish: Finish,
    ) -> Result<Option<Vec<Output>>>
    where
        State: Send + 'static,
        Output: Send + 'static,
        BuildState: Fn() -> State + Clone + Send + Sync + 'static,
        ConsumeBatch:
            FnMut(RecordBatch, &mut State) -> Result<Option<()>> + Clone + Send + Sync + 'static,
        Finish: Fn(State) -> Result<Option<Output>> + Clone + Send + Sync + 'static,
    {
        Ok(self
            .parquet_row_group_map_pruned_results(
                "row_group_map",
                path,
                batch_size,
                projection,
                pruning_predicates,
                row_group_chunk,
                build_state,
                consume_batch,
                finish,
            )
            .await?
            .map(|results| results.outputs))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn parquet_row_group_map_pruned_view<
        State,
        Output,
        BuildState,
        ConsumeBatch,
        Finish,
    >(
        &self,
        path: PathBuf,
        batch_size: usize,
        projection: Projection,
        pruning_predicates: Vec<Expr>,
        row_group_chunk: usize,
        build_state: BuildState,
        consume_batch: ConsumeBatch,
        finish: Finish,
    ) -> Result<Option<Vec<Output>>>
    where
        State: Send + 'static,
        Output: Send + 'static,
        BuildState: Fn() -> State + Clone + Send + Sync + 'static,
        ConsumeBatch: for<'a> FnMut(BatchView<'a>, &mut State) -> Result<Option<()>>
            + Clone
            + Send
            + Sync
            + 'static,
        Finish: Fn(State) -> Result<Option<Output>> + Clone + Send + Sync + 'static,
    {
        let source = self.plan_table_source(path.clone()).await?;
        if source.format != StorageFormat::Parquet || source.fragments.len() != 1 {
            return Ok(None);
        }
        let local_path = source.fragments[0].parquet_local_path()?.to_path_buf();
        let plan = plan_parquet_scan_tasks(
            &local_path,
            &projection,
            &pruning_predicates,
            &self.metadata_cache,
            self.object_store.as_ref(),
        )?;
        let row_groups = plan
            .tasks
            .into_iter()
            .map(|task| task.row_group)
            .collect::<Vec<_>>();
        if row_groups.is_empty() {
            return Ok(Some(Vec::new()));
        }
        let chunks = row_groups
            .chunks(row_group_chunk.max(1))
            .map(|chunk| chunk.to_vec())
            .collect::<Vec<_>>();
        let profile = scan_profile_enabled();
        let label = profile.then(|| parquet_map_profile_label(&local_path, &projection));
        let (sender, receiver) = mpsc::channel();
        for (index, row_groups) in chunks.iter().cloned().enumerate() {
            let sender = sender.clone();
            let path = local_path.clone();
            let projection = projection.clone();
            let label = label.clone();
            let metadata_cache = self.metadata_cache.clone();
            let file_cache = self.file_cache.clone();
            let object_store = self.object_store.clone();
            let build_state = build_state.clone();
            let consume_batch = consume_batch.clone();
            let finish = finish.clone();
            rayon::spawn(move || {
                let result = parquet_row_group_map_chunk_view(
                    path,
                    batch_size,
                    row_groups,
                    &projection,
                    &metadata_cache,
                    file_cache,
                    object_store.as_ref(),
                    build_state(),
                    consume_batch,
                    finish,
                    label.as_deref(),
                    index,
                );
                let _ = sender.send((index, result));
            });
        }
        drop(sender);

        let mut outputs = (0..chunks.len())
            .map(|_| None)
            .collect::<Vec<Option<Option<ParquetMapChunkResult<Output>>>>>();
        for _ in 0..chunks.len() {
            let (index, result) = receiver.recv().map_err(|_| {
                DodamError::UnsupportedSql("parquet row-group map worker stopped".to_string())
            })?;
            outputs[index] = Some(result?);
        }
        let mut results = Vec::with_capacity(outputs.len());
        let mut summary = ParquetMapChunkMetrics::default();
        for output in outputs {
            let Some(output) = output else {
                return Err(DodamError::UnsupportedSql(
                    "parquet row-group map result missing".to_string(),
                ));
            };
            let Some(output) = output else {
                return Ok(None);
            };
            summary.add(output.metrics);
            results.push(output.output);
        }
        log_parquet_map_summary("row_group_map_view", label.as_deref(), summary);
        Ok(Some(results))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn parquet_row_group_map_scan_view<
        State,
        Output,
        BuildState,
        ConsumeBatch,
        Finish,
    >(
        &self,
        path: PathBuf,
        batch_size: usize,
        projection: Projection,
        dictionary_columns: Vec<String>,
        pruning_predicates: Vec<Expr>,
        row_group_chunk: usize,
        build_state: BuildState,
        consume_batch: ConsumeBatch,
        finish: Finish,
    ) -> Result<Option<Vec<Output>>>
    where
        State: Send + 'static,
        Output: Send + 'static,
        BuildState: Fn() -> State + Clone + Send + Sync + 'static,
        ConsumeBatch: for<'a> FnMut(BatchView<'a>, &mut State) -> Result<Option<()>>
            + Clone
            + Send
            + Sync
            + 'static,
        Finish: Fn(State) -> Result<Option<Output>> + Clone + Send + Sync + 'static,
    {
        if dictionary_columns.is_empty() {
            return self
                .parquet_row_group_map_pruned_view(
                    path,
                    batch_size,
                    projection,
                    pruning_predicates,
                    row_group_chunk,
                    build_state,
                    consume_batch,
                    finish,
                )
                .await;
        }
        self.parquet_row_group_map_dictionary_columns_pruned_view(
            path,
            batch_size,
            projection,
            dictionary_columns,
            pruning_predicates,
            row_group_chunk,
            build_state,
            consume_batch,
            finish,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn parquet_row_group_map_pruned_results<State, Output, BuildState, ConsumeBatch, Finish>(
        &self,
        profile_kind: &str,
        path: PathBuf,
        batch_size: usize,
        projection: Projection,
        pruning_predicates: Vec<Expr>,
        row_group_chunk: usize,
        build_state: BuildState,
        consume_batch: ConsumeBatch,
        finish: Finish,
    ) -> Result<Option<ParquetMapResults<Output>>>
    where
        State: Send + 'static,
        Output: Send + 'static,
        BuildState: Fn() -> State + Clone + Send + Sync + 'static,
        ConsumeBatch:
            FnMut(RecordBatch, &mut State) -> Result<Option<()>> + Clone + Send + Sync + 'static,
        Finish: Fn(State) -> Result<Option<Output>> + Clone + Send + Sync + 'static,
    {
        let source = self.plan_table_source(path.clone()).await?;
        if source.format != StorageFormat::Parquet || source.fragments.len() != 1 {
            return Ok(None);
        }
        let local_path = source.fragments[0].parquet_local_path()?.to_path_buf();
        let plan = plan_parquet_scan_tasks(
            &local_path,
            &projection,
            &pruning_predicates,
            &self.metadata_cache,
            self.object_store.as_ref(),
        )?;
        let row_groups = plan
            .tasks
            .into_iter()
            .map(|task| task.row_group)
            .collect::<Vec<_>>();
        if row_groups.is_empty() {
            return Ok(Some(ParquetMapResults {
                outputs: Vec::new(),
                metrics: ParquetMapChunkMetrics::default(),
            }));
        }
        let chunks = row_groups
            .chunks(row_group_chunk.max(1))
            .map(|chunk| chunk.to_vec())
            .collect::<Vec<_>>();
        let profile = scan_profile_enabled();
        let label = profile.then(|| parquet_map_profile_label(&local_path, &projection));
        let (sender, receiver) = mpsc::channel();
        for (index, row_groups) in chunks.iter().cloned().enumerate() {
            let sender = sender.clone();
            let path = local_path.clone();
            let projection = projection.clone();
            let label = label.clone();
            let metadata_cache = self.metadata_cache.clone();
            let file_cache = self.file_cache.clone();
            let object_store = self.object_store.clone();
            let build_state = build_state.clone();
            let consume_batch = consume_batch.clone();
            let finish = finish.clone();
            rayon::spawn(move || {
                let result = parquet_row_group_map_chunk(
                    path,
                    batch_size,
                    row_groups,
                    &projection,
                    &metadata_cache,
                    file_cache,
                    object_store.as_ref(),
                    build_state(),
                    consume_batch,
                    finish,
                    label.as_deref(),
                    index,
                );
                let _ = sender.send((index, result));
            });
        }
        drop(sender);

        let mut outputs = (0..chunks.len())
            .map(|_| None)
            .collect::<Vec<Option<Option<ParquetMapChunkResult<Output>>>>>();
        for _ in 0..chunks.len() {
            let (index, result) = receiver.recv().map_err(|_| {
                DodamError::UnsupportedSql("parquet row-group map worker stopped".to_string())
            })?;
            outputs[index] = Some(result?);
        }
        let mut results = Vec::with_capacity(outputs.len());
        let mut summary = ParquetMapChunkMetrics::default();
        for output in outputs {
            let Some(output) = output else {
                return Err(DodamError::UnsupportedSql(
                    "parquet row-group map result missing".to_string(),
                ));
            };
            let Some(output) = output else {
                return Ok(None);
            };
            summary.add(output.metrics);
            results.push(output.output);
        }
        log_parquet_map_summary(profile_kind, label.as_deref(), summary);
        Ok(Some(ParquetMapResults {
            outputs: results,
            metrics: summary,
        }))
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn parquet_row_group_map_dictionary_columns<
        State,
        Output,
        BuildState,
        ConsumeBatch,
        Finish,
    >(
        &self,
        path: PathBuf,
        batch_size: usize,
        projection: Projection,
        dictionary_columns: Vec<String>,
        row_group_chunk: usize,
        build_state: BuildState,
        consume_batch: ConsumeBatch,
        finish: Finish,
    ) -> Result<Option<Vec<Output>>>
    where
        State: Send + 'static,
        Output: Send + 'static,
        BuildState: Fn() -> State + Clone + Send + Sync + 'static,
        ConsumeBatch:
            FnMut(RecordBatch, &mut State) -> Result<Option<()>> + Clone + Send + Sync + 'static,
        Finish: Fn(State) -> Result<Option<Output>> + Clone + Send + Sync + 'static,
    {
        self.parquet_row_group_map_dictionary_columns_pruned(
            path,
            batch_size,
            projection,
            dictionary_columns,
            Vec::new(),
            row_group_chunk,
            build_state,
            consume_batch,
            finish,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn parquet_row_group_map_dictionary_columns_pruned<
        State,
        Output,
        BuildState,
        ConsumeBatch,
        Finish,
    >(
        &self,
        path: PathBuf,
        batch_size: usize,
        projection: Projection,
        dictionary_columns: Vec<String>,
        pruning_predicates: Vec<Expr>,
        row_group_chunk: usize,
        build_state: BuildState,
        consume_batch: ConsumeBatch,
        finish: Finish,
    ) -> Result<Option<Vec<Output>>>
    where
        State: Send + 'static,
        Output: Send + 'static,
        BuildState: Fn() -> State + Clone + Send + Sync + 'static,
        ConsumeBatch:
            FnMut(RecordBatch, &mut State) -> Result<Option<()>> + Clone + Send + Sync + 'static,
        Finish: Fn(State) -> Result<Option<Output>> + Clone + Send + Sync + 'static,
    {
        let source = self.plan_table_source(path.clone()).await?;
        if source.format != StorageFormat::Parquet
            || source.fragments.len() != 1
            || dictionary_columns.is_empty()
        {
            return Ok(None);
        }
        let local_path = source.fragments[0].parquet_local_path()?.to_path_buf();
        let plan = plan_parquet_scan_tasks(
            &local_path,
            &projection,
            &pruning_predicates,
            &self.metadata_cache,
            self.object_store.as_ref(),
        )?;
        let row_groups = plan
            .tasks
            .into_iter()
            .map(|task| task.row_group)
            .collect::<Vec<_>>();
        if row_groups.is_empty() {
            return Ok(Some(Vec::new()));
        }
        let chunks = row_groups
            .chunks(row_group_chunk.max(1))
            .map(|chunk| chunk.to_vec())
            .collect::<Vec<_>>();
        let profile = scan_profile_enabled();
        let label = profile.then(|| parquet_map_profile_label(&local_path, &projection));
        let dictionary_columns = Arc::new(dictionary_columns);
        let (sender, receiver) = mpsc::channel();
        for (index, row_groups) in chunks.iter().cloned().enumerate() {
            let sender = sender.clone();
            let path = local_path.clone();
            let projection = projection.clone();
            let dictionary_columns = dictionary_columns.clone();
            let label = label.clone();
            let metadata_cache = self.metadata_cache.clone();
            let file_cache = self.file_cache.clone();
            let object_store = self.object_store.clone();
            let build_state = build_state.clone();
            let consume_batch = consume_batch.clone();
            let finish = finish.clone();
            rayon::spawn(move || {
                let result = parquet_row_group_map_dictionary_chunk(
                    path,
                    batch_size,
                    row_groups,
                    &projection,
                    dictionary_columns.as_ref(),
                    &metadata_cache,
                    file_cache,
                    object_store.as_ref(),
                    build_state(),
                    consume_batch,
                    finish,
                    label.as_deref(),
                    index,
                );
                let _ = sender.send((index, result));
            });
        }
        drop(sender);

        let mut outputs = (0..chunks.len())
            .map(|_| None)
            .collect::<Vec<Option<Option<ParquetMapChunkResult<Output>>>>>();
        for _ in 0..chunks.len() {
            let (index, result) = receiver.recv().map_err(|_| {
                DodamError::UnsupportedSql(
                    "parquet dictionary row-group map worker stopped".to_string(),
                )
            })?;
            outputs[index] = Some(result?);
        }
        let mut results = Vec::with_capacity(outputs.len());
        let mut summary = ParquetMapChunkMetrics::default();
        for output in outputs {
            let Some(output) = output else {
                return Err(DodamError::UnsupportedSql(
                    "parquet dictionary row-group map result missing".to_string(),
                ));
            };
            let Some(output) = output else {
                return Ok(None);
            };
            summary.add(output.metrics);
            results.push(output.output);
        }
        log_parquet_map_summary("dictionary_row_group_map", label.as_deref(), summary);
        Ok(Some(results))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn parquet_row_group_map_dictionary_columns_pruned_view<
        State,
        Output,
        BuildState,
        ConsumeBatch,
        Finish,
    >(
        &self,
        path: PathBuf,
        batch_size: usize,
        projection: Projection,
        dictionary_columns: Vec<String>,
        pruning_predicates: Vec<Expr>,
        row_group_chunk: usize,
        build_state: BuildState,
        consume_batch: ConsumeBatch,
        finish: Finish,
    ) -> Result<Option<Vec<Output>>>
    where
        State: Send + 'static,
        Output: Send + 'static,
        BuildState: Fn() -> State + Clone + Send + Sync + 'static,
        ConsumeBatch: for<'a> FnMut(BatchView<'a>, &mut State) -> Result<Option<()>>
            + Clone
            + Send
            + Sync
            + 'static,
        Finish: Fn(State) -> Result<Option<Output>> + Clone + Send + Sync + 'static,
    {
        let source = self.plan_table_source(path.clone()).await?;
        if source.format != StorageFormat::Parquet
            || source.fragments.len() != 1
            || dictionary_columns.is_empty()
        {
            return Ok(None);
        }
        let local_path = source.fragments[0].parquet_local_path()?.to_path_buf();
        let plan = plan_parquet_scan_tasks(
            &local_path,
            &projection,
            &pruning_predicates,
            &self.metadata_cache,
            self.object_store.as_ref(),
        )?;
        let row_groups = plan
            .tasks
            .into_iter()
            .map(|task| task.row_group)
            .collect::<Vec<_>>();
        if row_groups.is_empty() {
            return Ok(Some(Vec::new()));
        }
        let chunks = row_groups
            .chunks(row_group_chunk.max(1))
            .map(|chunk| chunk.to_vec())
            .collect::<Vec<_>>();
        let profile = scan_profile_enabled();
        let label = profile.then(|| parquet_map_profile_label(&local_path, &projection));
        let dictionary_columns = Arc::new(dictionary_columns);
        let (sender, receiver) = mpsc::channel();
        for (index, row_groups) in chunks.iter().cloned().enumerate() {
            let sender = sender.clone();
            let path = local_path.clone();
            let projection = projection.clone();
            let dictionary_columns = dictionary_columns.clone();
            let label = label.clone();
            let metadata_cache = self.metadata_cache.clone();
            let file_cache = self.file_cache.clone();
            let object_store = self.object_store.clone();
            let build_state = build_state.clone();
            let consume_batch = consume_batch.clone();
            let finish = finish.clone();
            rayon::spawn(move || {
                let result = parquet_row_group_map_dictionary_chunk_view(
                    path,
                    batch_size,
                    row_groups,
                    &projection,
                    dictionary_columns.as_ref(),
                    &metadata_cache,
                    file_cache,
                    object_store.as_ref(),
                    build_state(),
                    consume_batch,
                    finish,
                    label.as_deref(),
                    index,
                );
                let _ = sender.send((index, result));
            });
        }
        drop(sender);

        let mut outputs = (0..chunks.len())
            .map(|_| None)
            .collect::<Vec<Option<Option<ParquetMapChunkResult<Output>>>>>();
        for _ in 0..chunks.len() {
            let (index, result) = receiver.recv().map_err(|_| {
                DodamError::UnsupportedSql(
                    "parquet dictionary row-group map worker stopped".to_string(),
                )
            })?;
            outputs[index] = Some(result?);
        }
        let mut results = Vec::with_capacity(outputs.len());
        let mut summary = ParquetMapChunkMetrics::default();
        for output in outputs {
            let Some(output) = output else {
                return Err(DodamError::UnsupportedSql(
                    "parquet dictionary row-group map result missing".to_string(),
                ));
            };
            let Some(output) = output else {
                return Ok(None);
            };
            summary.add(output.metrics);
            results.push(output.output);
        }
        log_parquet_map_summary("dictionary_row_group_map_view", label.as_deref(), summary);
        Ok(Some(results))
    }

    pub async fn plan_parquet_scan(
        &self,
        path: PathBuf,
        batch_size: usize,
        limit: Option<usize>,
        projection: Projection,
        filter: Option<FilterExpr>,
        order_by: Option<SortKey>,
    ) -> Result<ScanPlan> {
        let source = self.plan_table_source(path).await?;
        self.plan_table_scan(source, batch_size, limit, projection, filter, order_by)
    }

    pub fn plan_table_scan(
        &self,
        source: TableScanSource,
        batch_size: usize,
        limit: Option<usize>,
        projection: Projection,
        filter: Option<FilterExpr>,
        order_by: Option<SortKey>,
    ) -> Result<ScanPlan> {
        self.plan_table_source_scan(source, batch_size, limit, projection, filter, order_by)
    }

    pub async fn explain_parquet_scan(
        &self,
        path: PathBuf,
        batch_size: usize,
        limit: Option<usize>,
        projection: Projection,
        filter: Option<FilterExpr>,
        order_by: Option<SortKey>,
    ) -> Result<String> {
        Ok(self
            .plan_parquet_scan(path, batch_size, limit, projection, filter, order_by)
            .await?
            .explain())
    }

    pub fn explain_table_scan(
        &self,
        source: TableScanSource,
        batch_size: usize,
        limit: Option<usize>,
        projection: Projection,
        filter: Option<FilterExpr>,
        order_by: Option<SortKey>,
    ) -> Result<String> {
        Ok(self
            .plan_table_scan(source, batch_size, limit, projection, filter, order_by)?
            .explain())
    }

    pub fn scan_table_source_batches(
        &self,
        source: TableScanSource,
        batch_size: usize,
        limit: Option<usize>,
        projection: Projection,
        filter: Option<FilterExpr>,
        order_by: Option<SortKey>,
    ) -> Result<SendableBatchStream> {
        let plan =
            self.plan_table_source_scan(source, batch_size, limit, projection, filter, order_by)?;
        self.execute_scan_plan(plan)
    }

    pub fn plan_table_source_scan(
        &self,
        source: TableScanSource,
        batch_size: usize,
        limit: Option<usize>,
        projection: Projection,
        filter: Option<FilterExpr>,
        order_by: Option<SortKey>,
    ) -> Result<ScanPlan> {
        self.plan_table_source_scan_with_options(
            source,
            ScanPlanOptions {
                batch_size,
                limit,
                projection,
                filter,
                order_by,
                distinct: false,
            },
        )
    }

    fn plan_table_source_scan_with_options(
        &self,
        source: TableScanSource,
        options: ScanPlanOptions,
    ) -> Result<ScanPlan> {
        let (source, filter) = prune_table_source_partitions(source, options.filter.clone());
        let logical = logical_scan_plan(
            &source,
            options.batch_size,
            options.limit,
            options.projection,
            filter,
            options.order_by,
            options.distinct,
        );
        let optimized = LogicalOptimizer.optimize(logical);
        if optimizer_trace_enabled() {
            eprintln!(
                "[dodam:optimizer] phase=logical rules={:?}",
                optimized.applied_rules
            );
        }
        let LogicalPlan::TableScan(scan) = optimized.plan else {
            return Err(DodamError::UnsupportedSql(
                "logical optimizer could not canonicalize scan plan".to_string(),
            ));
        };
        let scan_projection = if scan.distinct {
            scan_projection(&scan.projection, scan.filter.as_ref())
        } else {
            scan_projection_with_sort(
                &scan.projection,
                scan.filter.as_ref(),
                scan.order_by.as_ref(),
            )
        };
        let predicates = PredicateSet::new(scan.filter.clone());
        let estimated_bytes =
            self.estimate_scan_source_bytes(&source, &scan_projection, scan.filter.as_ref())?;
        let pushdown_predicates = predicates.pushdown().to_vec();
        let residual_filter = predicates.residual().cloned();
        let operators = scan_operators(
            scan.limit,
            scan.distinct,
            scan.filter.is_some(),
            scan.order_by.is_some(),
        );
        Ok(ScanPlan {
            source,
            batch_size: scan.batch_size,
            limit: scan.limit,
            output_projection: scan.projection,
            scan_projection,
            filter: scan.filter.clone(),
            residual_filter,
            pushdown_predicates,
            row_filter_predicates: Vec::new(),
            has_filter: scan.filter.is_some(),
            distinct: scan.distinct,
            order_by: scan.order_by,
            estimated_bytes,
            operators,
            preserve_order: false,
        })
    }

    pub async fn scan_parquet_ordered_batches(
        &self,
        path: PathBuf,
        batch_size: usize,
        limit: Option<usize>,
        projection: Projection,
        filter: Option<FilterExpr>,
        order_by: SortExpr,
    ) -> Result<SendableBatchStream> {
        self.scan_parquet_ordered_batches_by(
            path,
            batch_size,
            limit,
            projection,
            filter,
            SortKey::from(order_by),
        )
        .await
    }

    pub async fn scan_parquet_ordered_batches_by(
        &self,
        path: PathBuf,
        batch_size: usize,
        limit: Option<usize>,
        projection: Projection,
        filter: Option<FilterExpr>,
        order_by: SortKey,
    ) -> Result<SendableBatchStream> {
        let source = self.plan_table_source(path).await?;
        self.scan_table_source_batches(
            source,
            batch_size,
            limit,
            projection,
            filter,
            Some(order_by),
        )
    }

    pub async fn scan_parquet_distinct_batches(
        &self,
        path: PathBuf,
        batch_size: usize,
        limit: Option<usize>,
        projection: Projection,
        filter: Option<FilterExpr>,
        order_by: Option<SortKey>,
    ) -> Result<SendableBatchStream> {
        let source = self.plan_table_source(path).await?;
        let plan = self.plan_table_source_scan_with_options(
            source,
            ScanPlanOptions {
                batch_size,
                limit,
                projection,
                filter,
                order_by,
                distinct: true,
            },
        )?;
        self.execute_scan_plan(plan)
    }

    pub fn scan_table_distinct_batches(
        &self,
        source: TableScanSource,
        batch_size: usize,
        limit: Option<usize>,
        projection: Projection,
        filter: Option<FilterExpr>,
        order_by: Option<SortKey>,
    ) -> Result<SendableBatchStream> {
        let plan = self.plan_table_source_scan_with_options(
            source,
            ScanPlanOptions {
                batch_size,
                limit,
                projection,
                filter,
                order_by,
                distinct: true,
            },
        )?;
        self.execute_scan_plan(plan)
    }

    pub async fn plan_parquet_distinct_scan(
        &self,
        path: PathBuf,
        batch_size: usize,
        limit: Option<usize>,
        projection: Projection,
        filter: Option<FilterExpr>,
        order_by: Option<SortKey>,
    ) -> Result<ScanPlan> {
        let source = self.plan_table_source(path).await?;
        self.plan_table_source_scan_with_options(
            source,
            ScanPlanOptions {
                batch_size,
                limit,
                projection,
                filter,
                order_by,
                distinct: true,
            },
        )
    }

    pub fn plan_table_distinct_scan(
        &self,
        source: TableScanSource,
        batch_size: usize,
        limit: Option<usize>,
        projection: Projection,
        filter: Option<FilterExpr>,
        order_by: Option<SortKey>,
    ) -> Result<ScanPlan> {
        self.plan_table_source_scan_with_options(
            source,
            ScanPlanOptions {
                batch_size,
                limit,
                projection,
                filter,
                order_by,
                distinct: true,
            },
        )
    }

    pub async fn explain_parquet_distinct_scan(
        &self,
        path: PathBuf,
        batch_size: usize,
        limit: Option<usize>,
        projection: Projection,
        filter: Option<FilterExpr>,
        order_by: Option<SortKey>,
    ) -> Result<String> {
        Ok(self
            .plan_parquet_distinct_scan(path, batch_size, limit, projection, filter, order_by)
            .await?
            .explain())
    }

    pub async fn join_parquet_batches(
        &self,
        request: JoinParquetRequest,
    ) -> Result<SendableBatchStream> {
        self.execute_join_plan(self.plan_parquet_join(request).await?)
    }

    pub fn write_batches_to_sink(
        &self,
        stream: SendableBatchStream,
        sink: &mut dyn RecordBatchSink,
    ) -> Result<ScanPlanMetrics> {
        write_stream_to_sink(stream, sink)
    }

    pub fn execute_plan_to_sink(
        &self,
        plan: Box<dyn PhysicalPlan>,
        sink: &mut dyn RecordBatchSink,
    ) -> Result<ScanPlanMetrics> {
        plan.execute_to_sink(sink)
    }

    pub fn execute_physical_plan_node(
        &self,
        plan: PhysicalPlanNode,
    ) -> Result<SendableBatchStream> {
        self.build_physical_plan_node(plan)?.execute()
    }

    pub fn execute_task_plan(&self, task: TaskPlan) -> Result<SendableBatchStream> {
        let shuffle_store = LocalShuffleStore::new()?;
        self.execute_task_plan_with_shuffle(task, &shuffle_store)
    }

    fn execute_task_plan_with_shuffle(
        &self,
        task: TaskPlan,
        shuffle_store: &LocalShuffleStore,
    ) -> Result<SendableBatchStream> {
        let (stream, _) = self.execute_task_plan_with_shuffle_metrics(task, shuffle_store)?;
        Ok(stream)
    }

    fn execute_task_plan_with_shuffle_metrics(
        &self,
        task: TaskPlan,
        shuffle_store: &LocalShuffleStore,
    ) -> Result<(SendableBatchStream, LocalShuffleReadMetrics)> {
        let fragments = task
            .inputs
            .iter()
            .filter_map(|input| match input {
                TaskInput::ScanFragment(fragment) => Some(fragment.clone()),
                TaskInput::ShufflePartition { .. } => None,
            })
            .collect::<Vec<_>>();
        let shuffle_inputs = task
            .inputs
            .iter()
            .filter_map(|input| match input {
                TaskInput::ShufflePartition {
                    stage_id,
                    partition,
                } => Some((*stage_id, *partition)),
                TaskInput::ScanFragment(_) => None,
            })
            .collect::<Vec<_>>();
        let root = if fragments.is_empty() {
            task.root
        } else {
            physical_plan_with_scan_fragments(task.root, fragments)?
        };
        let root = if shuffle_inputs.is_empty() {
            root
        } else {
            let (root, read_metrics) =
                physical_plan_with_shuffle_inputs(root, &shuffle_inputs, shuffle_store)?;
            return Ok((self.execute_physical_plan_node(root)?, read_metrics));
        };
        Ok((
            self.execute_physical_plan_node(root)?,
            LocalShuffleReadMetrics::default(),
        ))
    }

    pub fn execute_execution_graph_locally(
        &self,
        graph: ExecutionGraphPlan,
    ) -> Result<Vec<SendableBatchStream>> {
        Ok(self
            .execute_execution_graph_locally_with_metrics(graph)?
            .streams)
    }

    pub fn execute_execution_graph_locally_with_metrics(
        &self,
        graph: ExecutionGraphPlan,
    ) -> Result<LocalExecutionGraphOutput> {
        self.execute_execution_graph_locally_with_options(graph, LocalExecutionOptions::default())
    }

    pub fn execute_execution_graph_locally_with_options(
        &self,
        graph: ExecutionGraphPlan,
        options: LocalExecutionOptions,
    ) -> Result<LocalExecutionGraphOutput> {
        let mut completed_stages = HashSet::new();
        let mut executed_stages = HashSet::new();
        let mut downstream_stages = HashSet::new();
        for stage in &graph.stages {
            downstream_stages.extend(stage.input_stages.iter().copied());
        }
        let mut shuffle_store =
            LocalShuffleStore::new_with_file_target_bytes(options.shuffle_file_target_bytes)?;
        let mut streams = Vec::new();
        let mut metrics = LocalExecutionGraphMetrics::default();
        let mut stage_metrics = Vec::new();

        while executed_stages.len() < graph.stages.len() {
            let mut progressed = false;
            for stage in &graph.stages {
                if executed_stages.contains(&stage.id)
                    || !stage
                        .input_stages
                        .iter()
                        .all(|input| completed_stages.contains(input))
                {
                    continue;
                }

                let stage_tasks = graph
                    .tasks
                    .iter()
                    .filter(|task| task.stage_id == stage.id)
                    .cloned()
                    .collect::<Vec<_>>();
                let mut current_stage_metrics = LocalStageExecutionMetrics {
                    stage_id: stage.id,
                    ..LocalStageExecutionMetrics::default()
                };
                for task in stage_tasks {
                    let stage_id = task.stage_id;
                    let partition = task.partition;
                    let started = Instant::now();
                    let (stream, read_metrics) =
                        self.execute_task_plan_with_shuffle_metrics(task, &shuffle_store)?;
                    let batches = stream.collect::<Result<Vec<_>>>()?;
                    let task_elapsed = started.elapsed();
                    current_stage_metrics.add_task_output(&batches, task_elapsed);
                    current_stage_metrics.add_shuffle_read(read_metrics);
                    metrics.task_execution_nanos = metrics
                        .task_execution_nanos
                        .saturating_add(elapsed_nanos(task_elapsed));
                    metrics.tasks_executed = metrics.tasks_executed.saturating_add(1);
                    metrics.task_output_batches =
                        metrics.task_output_batches.saturating_add(batches.len());
                    metrics.task_output_rows = metrics
                        .task_output_rows
                        .saturating_add(batches.iter().map(RecordBatch::num_rows).sum::<usize>());
                    metrics.shuffle_read_files = metrics
                        .shuffle_read_files
                        .saturating_add(read_metrics.files);
                    metrics.shuffle_read_batches = metrics
                        .shuffle_read_batches
                        .saturating_add(read_metrics.batches);
                    metrics.shuffle_read_rows =
                        metrics.shuffle_read_rows.saturating_add(read_metrics.rows);
                    metrics.shuffle_read_bytes = metrics
                        .shuffle_read_bytes
                        .saturating_add(read_metrics.bytes);
                    if downstream_stages.contains(&stage_id) {
                        match &stage.partitioning {
                            crate::plan::Partitioning::Hash { keys, partitions } => {
                                let repartition_started = Instant::now();
                                let partitioned =
                                    repartition_batches_by_hash(&batches, keys, *partitions)?;
                                let repartition_nanos =
                                    elapsed_nanos(repartition_started.elapsed());
                                metrics.shuffle_repartition_nanos = metrics
                                    .shuffle_repartition_nanos
                                    .saturating_add(repartition_nanos);
                                current_stage_metrics.shuffle_repartition_nanos =
                                    current_stage_metrics
                                        .shuffle_repartition_nanos
                                        .saturating_add(repartition_nanos);
                                for (shuffle_partition, partition_batches) in
                                    partitioned.into_iter().enumerate()
                                {
                                    let write_metrics = shuffle_store.write_partition(
                                        stage_id,
                                        shuffle_partition,
                                        &partition_batches,
                                    )?;
                                    metrics.add_shuffle_write(write_metrics);
                                    current_stage_metrics.add_shuffle_write(write_metrics);
                                }
                            }
                            crate::plan::Partitioning::Unknown
                            | crate::plan::Partitioning::Single
                            | crate::plan::Partitioning::RoundRobin { .. }
                            | crate::plan::Partitioning::FileRange { .. } => {
                                let write_metrics =
                                    shuffle_store.write_partition(stage_id, partition, &batches)?;
                                metrics.add_shuffle_write(write_metrics);
                                current_stage_metrics.add_shuffle_write(write_metrics);
                            }
                        }
                    } else {
                        streams.push(SendableBatchStream::from_batches(batches));
                    }
                }
                executed_stages.insert(stage.id);
                completed_stages.insert(stage.id);
                metrics.stages_executed = metrics.stages_executed.saturating_add(1);
                stage_metrics.push(current_stage_metrics);
                progressed = true;
            }

            if !progressed {
                return Err(DodamError::UnsupportedSql(
                    "execution graph contains cyclic or missing stage dependencies".to_string(),
                ));
            }
        }

        Ok(LocalExecutionGraphOutput {
            streams,
            metrics,
            stage_metrics,
        })
    }

    pub fn build_physical_plan_node(
        &self,
        plan: PhysicalPlanNode,
    ) -> Result<Box<dyn PhysicalPlan>> {
        let operator = plan.operator().clone();
        let execution = plan.execution_config().cloned();
        let children = plan.children().to_vec();
        match (operator, execution) {
            (
                PhysicalOperator::Scan,
                Some(PhysicalExecutionConfig::Scan {
                    fragments,
                    batch_size,
                    projection,
                    pushdown_predicates,
                }),
            ) => Ok(Box::new(ScanExec::new(
                fragments,
                batch_size,
                projection,
                pushdown_predicates,
                Vec::new(),
                self.metadata_cache.clone(),
                self.file_cache.clone(),
                self.object_store.clone(),
                false,
            ))),
            (PhysicalOperator::Memory, Some(PhysicalExecutionConfig::Memory { batches })) => {
                Ok(Box::new(MemoryExec::new(batches)))
            }
            (PhysicalOperator::Ipc, Some(PhysicalExecutionConfig::Ipc { files })) => {
                Ok(Box::new(IpcExec::new(files)))
            }
            (PhysicalOperator::Filter, Some(PhysicalExecutionConfig::Filter { filter })) => {
                let input = self.lower_single_child(children, "FilterExec")?;
                Ok(Box::new(FilterExec::new(input, filter)))
            }
            (
                PhysicalOperator::Projection,
                Some(PhysicalExecutionConfig::Projection { projection }),
            ) => {
                let input = self.lower_single_child(children, "ProjectionExec")?;
                Ok(Box::new(ProjectionExec::new(input, projection)))
            }
            (PhysicalOperator::Sort, Some(PhysicalExecutionConfig::Sort { order_by, limit })) => {
                let input = self.lower_single_child(children, "SortExec")?;
                Ok(Box::new(SortExec::new(input, order_by, limit)))
            }
            (PhysicalOperator::Limit, Some(PhysicalExecutionConfig::Limit { limit })) => {
                let input = self.lower_single_child(children, "LimitExec")?;
                Ok(Box::new(LimitExec::new(input, limit)))
            }
            (PhysicalOperator::Distinct, Some(PhysicalExecutionConfig::Distinct)) => {
                let input = self.lower_single_child(children, "DistinctExec")?;
                Ok(Box::new(DistinctExec::new(input)))
            }
            (
                PhysicalOperator::LocalFold,
                Some(PhysicalExecutionConfig::LocalFold {
                    group_by,
                    aggregates,
                }),
            ) => {
                let input = self.lower_single_child(children, "LocalFoldExec")?;
                Ok(Box::new(LocalFoldExec::new(input, group_by, aggregates)))
            }
            (
                PhysicalOperator::FinalMerge,
                Some(PhysicalExecutionConfig::FinalMerge {
                    group_by,
                    aggregates,
                }),
            ) => {
                let input = self.lower_single_child(children, "FinalMergeExec")?;
                Ok(Box::new(FinalMergeExec::new(input, group_by, aggregates)))
            }
            (
                PhysicalOperator::DirectPrimitiveFold,
                Some(PhysicalExecutionConfig::DirectPrimitiveFold {
                    path,
                    batch_size,
                    row_groups,
                    columns,
                    mode,
                }),
            ) => Ok(Box::new(DirectPrimitiveFoldExec::new(
                path,
                batch_size,
                row_groups,
                columns,
                mode,
                self.file_cache.clone(),
                self.object_store.clone(),
            ))),
            (
                PhysicalOperator::HashJoin,
                Some(PhysicalExecutionConfig::HashJoin {
                    left_keys,
                    right_keys,
                    left_prefix,
                    right_prefix,
                    build_side,
                    join_type,
                    output_projection,
                }),
            ) => {
                let (left, right) = self.lower_two_children(children, "JoinExec")?;
                Ok(Box::new(HashJoinExec::new(
                    left,
                    right,
                    left_keys,
                    right_keys,
                    left_prefix,
                    right_prefix,
                    build_side,
                    join_type,
                    output_projection,
                )))
            }
            (
                PhysicalOperator::PartitionedHashJoin,
                Some(PhysicalExecutionConfig::PartitionedHashJoin {
                    left_keys,
                    right_keys,
                    left_prefix,
                    right_prefix,
                    partitions,
                    memory_limit_bytes,
                    join_type,
                    output_projection,
                }),
            ) => {
                let (left, right) = self.lower_two_children(children, "PartitionedHashJoinExec")?;
                Ok(Box::new(PartitionedHashJoinExec::new(
                    left,
                    right,
                    left_keys,
                    right_keys,
                    left_prefix,
                    right_prefix,
                    PartitionedHashJoinOptions {
                        partitions,
                        memory_limit_bytes,
                        join_type,
                        output_projection,
                    },
                )))
            }
            (
                PhysicalOperator::SortMergeJoin,
                Some(PhysicalExecutionConfig::SortMergeJoin {
                    left_key,
                    right_key,
                    left_prefix,
                    right_prefix,
                }),
            ) => {
                let (left, right) = self.lower_two_children(children, "SortMergeJoinExec")?;
                Ok(Box::new(SortMergeJoinExec::new(
                    left,
                    right,
                    left_key,
                    right_key,
                    left_prefix,
                    right_prefix,
                )))
            }
            (operator, _) => Err(DodamError::UnsupportedSql(format!(
                "cannot lower declarative physical operator {operator:?} to local executor"
            ))),
        }
    }

    fn lower_single_child(
        &self,
        children: Vec<PhysicalPlanNode>,
        operator: &str,
    ) -> Result<Box<dyn PhysicalPlan>> {
        let [child] = children.try_into().map_err(|_| {
            DodamError::UnsupportedSql(format!("{operator} expects exactly one child"))
        })?;
        self.build_physical_plan_node(child)
    }

    fn lower_two_children(
        &self,
        children: Vec<PhysicalPlanNode>,
        operator: &str,
    ) -> Result<(Box<dyn PhysicalPlan>, Box<dyn PhysicalPlan>)> {
        let [left, right] = children.try_into().map_err(|_| {
            DodamError::UnsupportedSql(format!("{operator} expects exactly two children"))
        })?;
        Ok((
            self.build_physical_plan_node(left)?,
            self.build_physical_plan_node(right)?,
        ))
    }

    pub async fn explain_join_parquet(&self, request: JoinParquetRequest) -> Result<String> {
        Ok(self.plan_parquet_join(request).await?.explain())
    }

    pub async fn plan_parquet_join(&self, request: JoinParquetRequest) -> Result<JoinPlan> {
        let left_source = self.plan_table_source(request.left_path.clone()).await?;
        let right_source = self.plan_table_source(request.right_path.clone()).await?;
        self.plan_table_join(JoinTableRequest {
            left: left_source,
            right: right_source,
            batch_size: request.batch_size,
            left_keys: request.left_keys,
            right_keys: request.right_keys,
            left_prefix: request.left_prefix,
            right_prefix: request.right_prefix,
            left_projection: request.left_projection,
            right_projection: request.right_projection,
            left_filter: request.left_filter,
            right_filter: request.right_filter,
            output_projection: request.output_projection,
            join_memory_limit_bytes: request.join_memory_limit_bytes,
            join_algorithm: request.join_algorithm,
            join_type: request.join_type,
        })
    }

    pub fn join_tables(&self, request: JoinTableRequest) -> Result<SendableBatchStream> {
        self.execute_join_plan(self.plan_table_join(request)?)
    }

    pub fn write_join_plan_to_sink(
        &self,
        plan: JoinPlan,
        sink: &mut dyn RecordBatchSink,
    ) -> Result<ScanPlanMetrics> {
        self.build_physical_join_plan(plan).execute_to_sink(sink)
    }

    pub fn explain_table_join(&self, request: JoinTableRequest) -> Result<String> {
        Ok(self.plan_table_join(request)?.explain())
    }

    pub fn plan_table_join(&self, request: JoinTableRequest) -> Result<JoinPlan> {
        let left_scan = self.plan_table_source_scan(
            request.left,
            request.batch_size,
            None,
            request.left_projection.clone(),
            request.left_filter.clone(),
            None,
        )?;
        let right_scan = self.plan_table_source_scan(
            request.right,
            request.batch_size,
            None,
            request.right_projection.clone(),
            request.right_filter.clone(),
            None,
        )?;
        let strategy = choose_join_strategy(JoinCostInput {
            left_estimated_bytes: left_scan.estimated_bytes,
            right_estimated_bytes: right_scan.estimated_bytes,
            memory_limit_bytes: request.join_memory_limit_bytes,
            requested_algorithm: request.join_algorithm,
            join_type: request.join_type,
            left_keys: request.left_keys.len(),
            right_keys: request.right_keys.len(),
        });

        Ok(JoinPlan {
            request: JoinTablePlanRequest {
                batch_size: request.batch_size,
                left_keys: request.left_keys,
                right_keys: request.right_keys,
                left_prefix: request.left_prefix,
                right_prefix: request.right_prefix,
                left_projection: request.left_projection,
                right_projection: request.right_projection,
                left_filter: request.left_filter,
                right_filter: request.right_filter,
                output_projection: request.output_projection,
                join_memory_limit_bytes: request.join_memory_limit_bytes,
                join_algorithm: request.join_algorithm,
                join_type: request.join_type,
            },
            left_scan,
            right_scan,
            strategy,
        })
    }

    fn execute_join_plan(&self, plan: JoinPlan) -> Result<SendableBatchStream> {
        self.build_physical_join_plan(plan).execute()
    }

    fn build_physical_join_plan(&self, plan: JoinPlan) -> Box<dyn PhysicalPlan> {
        let left = self.build_physical_scan_plan(plan.left_scan);
        let right = self.build_physical_scan_plan(plan.right_scan);

        match plan.strategy {
            JoinExecutionStrategy::SortMerge => Box::new(SortMergeJoinExec::new(
                left,
                right,
                plan.request.left_keys[0].clone(),
                plan.request.right_keys[0].clone(),
                plan.request.left_prefix,
                plan.request.right_prefix,
            )),
            JoinExecutionStrategy::PartitionedHash {
                partitions,
                memory_limit_bytes,
            } => Box::new(PartitionedHashJoinExec::new(
                left,
                right,
                plan.request.left_keys,
                plan.request.right_keys,
                plan.request.left_prefix,
                plan.request.right_prefix,
                PartitionedHashJoinOptions {
                    partitions,
                    memory_limit_bytes,
                    join_type: plan.request.join_type,
                    output_projection: plan.request.output_projection,
                },
            )),
            JoinExecutionStrategy::Hash { build_side } => Box::new(HashJoinExec::new(
                left,
                right,
                plan.request.left_keys,
                plan.request.right_keys,
                plan.request.left_prefix,
                plan.request.right_prefix,
                build_side,
                plan.request.join_type,
                plan.request.output_projection,
            )),
        }
    }

    pub async fn aggregate_parquet(
        &self,
        path: PathBuf,
        batch_size: usize,
        aggregates: Vec<AggregateExpr>,
        filter: Option<FilterExpr>,
    ) -> Result<AggregateMetrics> {
        if let Some(metrics) = self
            .try_aggregate_parquet_fused(
                path.clone(),
                batch_size,
                aggregates.clone(),
                Vec::new(),
                filter.clone(),
            )
            .await?
        {
            return Ok(metrics);
        }
        let source = self.plan_table_source(path).await?;
        self.aggregate_table(source, batch_size, aggregates, Vec::new(), filter)
    }

    pub fn aggregate_table(
        &self,
        source: TableScanSource,
        batch_size: usize,
        aggregates: Vec<AggregateExpr>,
        group_by: Vec<String>,
        filter: Option<FilterExpr>,
    ) -> Result<AggregateMetrics> {
        let plan = self.plan_table_aggregate(source, batch_size, aggregates, group_by, filter)?;
        self.execute_aggregate_plan(plan)
    }

    pub async fn aggregate_parquet_grouped(
        &self,
        path: PathBuf,
        batch_size: usize,
        aggregates: Vec<AggregateExpr>,
        group_by: Vec<String>,
        filter: Option<FilterExpr>,
    ) -> Result<AggregateMetrics> {
        if let Some(metrics) = self
            .try_aggregate_parquet_fused(
                path.clone(),
                batch_size,
                aggregates.clone(),
                group_by.clone(),
                filter.clone(),
            )
            .await?
        {
            return Ok(metrics);
        }
        let source = self.plan_table_source(path).await?;
        self.aggregate_table(source, batch_size, aggregates, group_by, filter)
    }

    async fn try_aggregate_parquet_fused(
        &self,
        path: PathBuf,
        batch_size: usize,
        aggregates: Vec<AggregateExpr>,
        group_by: Vec<String>,
        filter: Option<FilterExpr>,
    ) -> Result<Option<AggregateMetrics>> {
        if !fused_parquet_aggregate_enabled() || !can_merge_partial_aggregates(&aggregates) {
            return Ok(None);
        }
        if let Some(filter) = filter {
            if let Some(metrics) = self
                .try_direct_primitive_count_sum_min_max_aggregate(
                    path.clone(),
                    batch_size,
                    &aggregates,
                    &group_by,
                    &filter,
                )
                .await?
            {
                return Ok(Some(metrics));
            }
            if let Some(metrics) = self
                .try_direct_i32_utf8_count_sum_aggregate(
                    path.clone(),
                    batch_size,
                    &aggregates,
                    &group_by,
                    &filter,
                )
                .await?
            {
                return Ok(Some(metrics));
            }
            if let Some(metrics) = self
                .try_filtered_count_sum_aggregate(
                    path.clone(),
                    batch_size,
                    &aggregates,
                    &group_by,
                    &filter,
                )
                .await?
            {
                return Ok(Some(metrics));
            }
            if let Some(metrics) = self
                .try_filtered_dictionary_count_sum_aggregate(
                    path.clone(),
                    batch_size,
                    &aggregates,
                    &group_by,
                    &filter,
                )
                .await?
            {
                return Ok(Some(metrics));
            }
            return self
                .try_late_materialized_parquet_aggregate(
                    path, batch_size, aggregates, group_by, filter,
                )
                .await;
        }
        if let Some(metrics) = self
            .try_direct_primitive_count_sum_aggregate(
                path.clone(),
                batch_size,
                &aggregates,
                &group_by,
            )
            .await?
        {
            return Ok(Some(metrics));
        }
        let projection = aggregate_projection(&aggregates, &group_by);
        if matches!(projection, Projection::All) {
            return Ok(None);
        }
        let Some(partials) = self
            .parquet_row_group_map(
                path,
                batch_size,
                projection,
                fused_parquet_aggregate_row_group_chunk(),
                Vec::<RecordBatch>::new,
                |batch, batches| {
                    batches.push(batch);
                    Ok(Some(()))
                },
                {
                    let aggregates = aggregates.clone();
                    let group_by = group_by.clone();
                    move |batches| {
                        if batches.is_empty() {
                            return Ok(None);
                        }
                        let stream = Box::new(MemoryExec::new(batches)).execute()?;
                        let metrics = if group_by.is_empty() {
                            collect_aggregates(stream, 1, &aggregates)?
                        } else {
                            collect_grouped_aggregates(stream, 1, &group_by, &aggregates)?
                        };
                        Ok(Some(vec![metrics]))
                    }
                },
            )
            .await?
        else {
            return Ok(None);
        };
        let partials = partials.into_iter().flatten().collect::<Vec<_>>();
        if partials.is_empty() {
            return Ok(None);
        }
        let started = Instant::now();
        let metrics = merge_partial_aggregate_metrics(partials, 1, &group_by, &aggregates)?;
        log_aggregate_profile("fused_parquet_aggregate_merge", &metrics, started.elapsed());
        Ok(Some(metrics))
    }

    async fn try_direct_primitive_count_sum_min_max_aggregate(
        &self,
        path: PathBuf,
        batch_size: usize,
        aggregates: &[AggregateExpr],
        group_by: &[String],
        filter: &FilterExpr,
    ) -> Result<Option<AggregateMetrics>> {
        let Some(shape) =
            DirectCountSumMinMaxShape::try_new(aggregates, group_by, filter, self, &path)?
        else {
            return Ok(None);
        };
        let source = self.plan_table_source(path.clone()).await?;
        let Some(node) = self.try_build_direct_primitive_fold_node(
            &source,
            batch_size,
            aggregates,
            group_by,
            Some(filter),
        )?
        else {
            return Ok(None);
        };
        let local_path = source.fragments[0].parquet_local_path()?.to_path_buf();
        let row_groups = direct_primitive_fold_row_groups(&node)?;
        let started = Instant::now();
        let scan_result = self.try_selected_payload_count_sum_min_max_metrics(
            &local_path,
            batch_size,
            &row_groups,
            &shape,
            aggregates,
        )?;
        let mut metrics = if let Some((state, scan_metrics)) = scan_result {
            let mut metrics = state.finish();
            metrics.fragments = 1;
            metrics.batches = scan_metrics.batches;
            metrics.rows = scan_metrics.rows;
            metrics
        } else {
            let Some(metrics) = self.try_execute_direct_primitive_fold_metrics(node)? else {
                return Ok(None);
            };
            metrics
        };
        metrics.aggregate_nanos = elapsed_nanos(started.elapsed());
        log_aggregate_profile(
            "direct_primitive_count_sum_min_max",
            &metrics,
            started.elapsed(),
        );
        Ok(Some(metrics))
    }

    async fn try_direct_primitive_count_sum_aggregate(
        &self,
        path: PathBuf,
        batch_size: usize,
        aggregates: &[AggregateExpr],
        group_by: &[String],
    ) -> Result<Option<AggregateMetrics>> {
        let source = self.plan_table_source(path.clone()).await?;
        let Some(node) = self.try_build_direct_primitive_fold_node(
            &source, batch_size, aggregates, group_by, None,
        )?
        else {
            return Ok(None);
        };
        let started = Instant::now();
        let Some(mut metrics) = self.try_execute_direct_primitive_fold_metrics(node)? else {
            return Ok(None);
        };
        metrics.aggregate_nanos = elapsed_nanos(started.elapsed());
        log_aggregate_profile("direct_primitive_count_sum", &metrics, started.elapsed());
        Ok(Some(metrics))
    }

    async fn try_direct_i32_utf8_count_sum_aggregate(
        &self,
        path: PathBuf,
        batch_size: usize,
        aggregates: &[AggregateExpr],
        group_by: &[String],
        filter: &FilterExpr,
    ) -> Result<Option<AggregateMetrics>> {
        if !direct_i32_utf8_count_sum_aggregate_enabled() {
            return Ok(None);
        }
        let Some(shape) = direct_i32_utf8_count_sum_shape(aggregates, group_by, filter)? else {
            return Ok(None);
        };
        let source = self.plan_table_source(path).await?;
        if source.format != StorageFormat::Parquet || source.fragments.len() != 1 {
            return Ok(None);
        }
        let local_path = source.fragments[0].parquet_local_path()?.to_path_buf();
        let projection = Projection::Columns(vec![
            shape.predicate_column.clone(),
            shape.sum_column.clone(),
            shape.group_column.clone(),
        ]);
        let predicates = PredicateSet::new(Some(filter.clone()));
        let row_groups =
            self.direct_primitive_row_groups(&local_path, &projection, predicates.pushdown())?;
        let mut state = DirectUtf8CountSumState::default();
        let started = Instant::now();
        let scan_result = self.scan_parquet_i32_i64_dictionary_id_columns(
            &local_path,
            batch_size,
            &row_groups,
            [
                shape.predicate_column.as_str(),
                shape.sum_column.as_str(),
                shape.group_column.as_str(),
            ],
            |predicate_values, sum_values, group_def_levels, group_ids, dictionary| {
                state.consume_dictionary_ids(
                    predicate_values,
                    sum_values,
                    group_def_levels,
                    group_ids,
                    dictionary,
                    &shape,
                );
                Ok(Some(()))
            },
        )?;
        let scan_metrics = if let Some(scan_metrics) = scan_result {
            scan_metrics
        } else {
            let Some(scan_metrics) = self.scan_parquet_i32_i64_byte_array_columns(
                &local_path,
                batch_size,
                &row_groups,
                [
                    shape.predicate_column.as_str(),
                    shape.sum_column.as_str(),
                    shape.group_column.as_str(),
                ],
                |predicate_values, sum_values, group_def_levels, group_values| {
                    state.consume(
                        predicate_values,
                        sum_values,
                        group_def_levels,
                        group_values,
                        &shape,
                    );
                    Ok(Some(()))
                },
            )?
            else {
                return Ok(None);
            };
            scan_metrics
        };
        let mut metrics = state.finish(1, aggregates[0].clone(), aggregates[1].clone())?;
        metrics.batches = scan_metrics.batches;
        metrics.aggregate_nanos = elapsed_nanos(started.elapsed());
        log_aggregate_profile(
            "direct_i32_utf8_count_sum_aggregate",
            &metrics,
            started.elapsed(),
        );
        Ok(Some(metrics))
    }

    async fn try_filtered_count_sum_aggregate(
        &self,
        path: PathBuf,
        batch_size: usize,
        aggregates: &[AggregateExpr],
        group_by: &[String],
        filter: &FilterExpr,
    ) -> Result<Option<AggregateMetrics>> {
        if !filtered_count_sum_aggregate_enabled() {
            return Ok(None);
        }
        if SingleKeyCountSumBatchAccumulator::try_new(1, group_by, aggregates).is_none() {
            return Ok(None);
        }
        let column_read_plan = aggregate_column_read_plan(aggregates, group_by, Some(filter));
        if matches!(column_read_plan.scan_projection, Projection::All) {
            return Ok(None);
        }
        let predicates = PredicateSet::new(Some(filter.clone()));
        let Some(partials) = self
            .parquet_row_group_map_pruned_view(
                path,
                batch_size,
                column_read_plan.scan_projection.clone(),
                predicates.pushdown().to_vec(),
                fused_parquet_aggregate_row_group_chunk(),
                {
                    let aggregates = aggregates.to_vec();
                    let group_by = group_by.to_vec();
                    move || {
                        SingleKeyCountSumBatchAccumulator::try_new(1, &group_by, &aggregates)
                            .expect("filtered count/sum aggregate shape checked")
                    }
                },
                {
                    let filter = filter.clone();
                    move |view, state| {
                        let Some(batch) = view.try_record_batch() else {
                            return Err(DodamError::UnsupportedSql(
                                "filtered count/sum aggregate requires RecordBatch".to_string(),
                            ));
                        };
                        let mask = evaluate_filter_mask(batch, &filter)?;
                        if !state.consume_filtered_batch(batch, &mask)? {
                            return Ok(None);
                        }
                        Ok(Some(()))
                    }
                },
                |state| Ok(Some(vec![state.finish()])),
            )
            .await?
        else {
            return Ok(None);
        };
        let partials = partials.into_iter().flatten().collect::<Vec<_>>();
        if partials.is_empty() {
            return Ok(None);
        }
        let started = Instant::now();
        let metrics = merge_partial_aggregate_metrics(partials, 1, group_by, aggregates)?;
        log_aggregate_profile("filtered_count_sum_aggregate", &metrics, started.elapsed());
        Ok(Some(metrics))
    }

    async fn try_filtered_dictionary_count_sum_aggregate(
        &self,
        path: PathBuf,
        batch_size: usize,
        aggregates: &[AggregateExpr],
        group_by: &[String],
        filter: &FilterExpr,
    ) -> Result<Option<AggregateMetrics>> {
        if !filtered_dictionary_aggregate_enabled() {
            return Ok(None);
        }
        if SingleKeyCountSumBatchAccumulator::try_new(1, group_by, aggregates).is_none() {
            return Ok(None);
        }
        let dictionary_columns =
            late_materialized_aggregate_dictionary_columns(aggregates, group_by);
        if dictionary_columns.is_empty() {
            return Ok(None);
        }
        let column_read_plan = aggregate_column_read_plan(aggregates, group_by, Some(filter));
        if matches!(column_read_plan.scan_projection, Projection::All) {
            return Ok(None);
        }
        let predicates = PredicateSet::new(Some(filter.clone()));
        let Some(partials) = self
            .parquet_row_group_map_dictionary_columns_pruned_view(
                path,
                batch_size,
                column_read_plan.scan_projection.clone(),
                dictionary_columns,
                predicates.pushdown().to_vec(),
                filtered_dictionary_aggregate_row_group_chunk(),
                {
                    let aggregates = aggregates.to_vec();
                    let group_by = group_by.to_vec();
                    move || {
                        SingleKeyCountSumBatchAccumulator::try_new(1, &group_by, &aggregates)
                            .expect("filtered dictionary aggregate shape checked")
                    }
                },
                {
                    let filter = filter.clone();
                    move |view, state| {
                        let Some(batch) = view.try_record_batch() else {
                            return Err(DodamError::UnsupportedSql(
                                "filtered dictionary aggregate requires RecordBatch".to_string(),
                            ));
                        };
                        let mask = evaluate_filter_mask(batch, &filter)?;
                        if !state.consume_filtered_batch(batch, &mask)? {
                            return Ok(None);
                        }
                        Ok(Some(()))
                    }
                },
                |state| Ok(Some(vec![state.finish()])),
            )
            .await?
        else {
            return Ok(None);
        };
        let partials = partials.into_iter().flatten().collect::<Vec<_>>();
        if partials.is_empty() {
            return Ok(None);
        }
        let started = Instant::now();
        let metrics = merge_partial_aggregate_metrics(partials, 1, group_by, aggregates)?;
        log_aggregate_profile(
            "filtered_dictionary_count_sum_aggregate",
            &metrics,
            started.elapsed(),
        );
        Ok(Some(metrics))
    }

    fn try_build_direct_primitive_fold_node(
        &self,
        source: &TableScanSource,
        batch_size: usize,
        aggregates: &[AggregateExpr],
        group_by: &[String],
        filter: Option<&FilterExpr>,
    ) -> Result<Option<PhysicalPlanNode>> {
        if source.format != StorageFormat::Parquet || source.fragments.len() != 1 {
            log_optimizer_rule_trace(
                "direct_primitive_fold",
                "reject",
                "requires single-fragment parquet source",
            );
            return Ok(None);
        }
        let local_path = source.fragments[0].parquet_local_path()?.to_path_buf();
        if let Some(filter) = filter {
            let Some(shape) = DirectCountSumMinMaxShape::try_new(
                aggregates,
                group_by,
                filter,
                self,
                &local_path,
            )?
            else {
                log_optimizer_rule_trace(
                    "direct_primitive_fold",
                    "reject",
                    "unsupported filtered count/sum/min/max shape",
                );
                return Ok(None);
            };
            let projection = Projection::Columns(shape.projection_columns());
            let predicates = PredicateSet::new(Some(filter.clone()));
            let row_groups =
                self.direct_primitive_row_groups(&local_path, &projection, predicates.pushdown())?;
            let decimal_min = option_i128_to_i64(shape.filter.decimal_min)?;
            let decimal_max = option_i128_to_i64(shape.filter.decimal_max)?;
            let columns_attr = format!("[{}]", shape.projection_columns().join(","));
            let max_decimal = matches!(shape.max_kind, CountSumMinMaxMaxKind::Decimal128 { .. });
            let node = if let Some(second_key_column) = &shape.second_key_column {
                if second_key_column != &shape.filter_date_column {
                    log_optimizer_rule_trace(
                        "direct_primitive_fold",
                        "reject",
                        "two-key count/sum/min/max requires Date32 group key to be filter date",
                    );
                    return Ok(None);
                }
                PhysicalPlanNode::new("DirectPrimitiveFoldExec")
                    .attr("mode", "two_key_count_sum_min_max")
                    .attr("rule", "direct_primitive_fold")
                    .attr(
                        "group_by",
                        format!("{},{}", shape.key_column, second_key_column),
                    )
                    .attr("row_groups", row_groups.len())
                    .attr("columns", columns_attr)
                    .execution(PhysicalExecutionConfig::DirectPrimitiveFold {
                        path: local_path,
                        batch_size,
                        row_groups,
                        columns: vec![
                            (
                                shape.key_column.clone(),
                                shape.key_type.column_type_descriptor().to_string(),
                            ),
                            (second_key_column.clone(), "date32".to_string()),
                            (shape.sum_column.clone(), "i64".to_string()),
                            (
                                shape.min_decimal_column.clone(),
                                format!(
                                    "decimal128_i64_raw:{}:{}",
                                    shape.decimal_precision, shape.decimal_scale
                                ),
                            ),
                        ],
                        mode: DirectPrimitiveFoldMode::TwoKeyCountSumMinMax {
                            first_group_by: shape.key_column.clone(),
                            first_key_type: shape.key_type.column_type_descriptor().to_string(),
                            second_group_by: second_key_column.clone(),
                            aggregates: aggregates.to_vec(),
                            decimal_precision: shape.decimal_precision,
                            decimal_scale: shape.decimal_scale,
                            max_decimal,
                            decimal_min,
                            decimal_max,
                            date_min: shape.filter.date_min,
                            date_max: shape.filter.date_max,
                        },
                    })
            } else {
                PhysicalPlanNode::new("DirectPrimitiveFoldExec")
                    .attr("mode", "single_key_count_sum_min_max")
                    .attr("rule", "direct_primitive_fold")
                    .attr("group_by", shape.key_column.clone())
                    .attr("row_groups", row_groups.len())
                    .attr("columns", columns_attr)
                    .execution(PhysicalExecutionConfig::DirectPrimitiveFold {
                        path: local_path,
                        batch_size,
                        row_groups,
                        columns: vec![
                            (
                                shape.key_column.clone(),
                                shape.key_type.column_type_descriptor().to_string(),
                            ),
                            (shape.sum_column.clone(), "i64".to_string()),
                            (
                                shape.min_decimal_column.clone(),
                                format!(
                                    "decimal128_i64_raw:{}:{}",
                                    shape.decimal_precision, shape.decimal_scale
                                ),
                            ),
                            (shape.filter_date_column.clone(), "date32".to_string()),
                        ],
                        mode: DirectPrimitiveFoldMode::SingleKeyCountSumMinMax {
                            group_by: shape.key_column.clone(),
                            key_type: shape.key_type.column_type_descriptor().to_string(),
                            aggregates: aggregates.to_vec(),
                            decimal_precision: shape.decimal_precision,
                            decimal_scale: shape.decimal_scale,
                            max_decimal,
                            decimal_min,
                            decimal_max,
                            date_min: shape.filter.date_min,
                            date_max: shape.filter.date_max,
                        },
                    })
            };
            log_optimizer_rule_trace(
                "direct_primitive_fold",
                "accept",
                "count/sum/min/max primitive aggregate",
            );
            return Ok(Some(node));
        }

        let Some(shape) = DirectCountSumShape::try_new(aggregates, group_by, self, &local_path)?
        else {
            log_optimizer_rule_trace(
                "direct_primitive_fold",
                "reject",
                "unsupported count/sum shape",
            );
            return Ok(None);
        };
        let projection = Projection::Columns(shape.projection_columns());
        let row_groups = self.direct_primitive_row_groups(&local_path, &projection, &[])?;
        let node = PhysicalPlanNode::new("DirectPrimitiveFoldExec")
            .attr("mode", "single_key_count_sum")
            .attr("rule", "direct_primitive_fold")
            .attr("group_by", shape.key_column.clone())
            .attr("row_groups", row_groups.len())
            .attr(
                "columns",
                format!("[{}]", shape.projection_columns().join(",")),
            )
            .execution(PhysicalExecutionConfig::DirectPrimitiveFold {
                path: local_path,
                batch_size,
                row_groups,
                columns: vec![
                    (
                        shape.key_column.clone(),
                        shape.key_type.column_type_descriptor().to_string(),
                    ),
                    (shape.sum_column.clone(), "i64".to_string()),
                ],
                mode: DirectPrimitiveFoldMode::SingleKeyCountSum {
                    group_by: shape.key_column.clone(),
                    key_type: shape.key_type.column_type_descriptor().to_string(),
                    count: shape.count_expr.clone(),
                    sum: aggregates[1].clone(),
                },
            });
        log_optimizer_rule_trace(
            "direct_primitive_fold",
            "accept",
            "single-key count/sum primitive aggregate",
        );
        Ok(Some(node))
    }

    fn try_selected_payload_count_sum_min_max_metrics(
        &self,
        local_path: &Path,
        batch_size: usize,
        row_groups: &[usize],
        shape: &DirectCountSumMinMaxShape,
        aggregates: &[AggregateExpr],
    ) -> Result<
        Option<(
            SingleKeyCountSumMinMaxVectorState,
            DirectPrimitiveColumnScanMetrics,
        )>,
    > {
        if !direct_selection_fold_enabled()
            || !shape.is_single_key()
            || !matches!(shape.key_type, DirectPrimitiveKeyType::I32)
        {
            return Ok(None);
        }
        let columns = [
            shape.key_column.as_str(),
            shape.sum_column.as_str(),
            shape.min_decimal_column.as_str(),
            shape.filter_date_column.as_str(),
        ];
        if decimal_date_selected_typed_agg_enabled() {
            return self.scan_parquet_i32_i64_decimal_i32_selected_typed_fold(
                local_path,
                batch_size,
                row_groups,
                columns,
                shape.decimal_precision,
                shape.decimal_scale,
                shape.filter,
                || {
                    SingleKeyCountSumMinMaxVectorState::new_i32_with_max_kind(
                        aggregates.to_vec(),
                        shape.decimal_precision,
                        shape.decimal_scale,
                        shape.max_kind,
                    )
                },
                |state, batch| {
                    state.consume_i32_i64_decimal_date_slices(
                        batch.keys,
                        batch.sums,
                        batch.decimals,
                        batch.dates,
                        &shape.filter,
                        batch.predicate_applied,
                    )
                },
                |state, partial| state.merge(partial),
            );
        }
        self.scan_parquet_i32_i64_decimal_i32_selected_batch_fold(
            local_path,
            batch_size,
            row_groups,
            columns,
            shape.decimal_precision,
            shape.decimal_scale,
            shape.filter,
            || {
                SingleKeyCountSumMinMaxVectorState::new_i32_with_max_kind(
                    aggregates.to_vec(),
                    shape.decimal_precision,
                    shape.decimal_scale,
                    shape.max_kind,
                )
            },
            |state, batch| state.consume_i32_i64_decimal_date_batch(batch, &shape.filter),
            |state, partial| state.merge(partial),
        )
    }

    fn direct_primitive_row_groups(
        &self,
        local_path: &Path,
        projection: &Projection,
        pushdown_predicates: &[Expr],
    ) -> Result<Vec<usize>> {
        Ok(plan_parquet_scan_tasks(
            local_path,
            projection,
            pushdown_predicates,
            &self.metadata_cache,
            self.object_store.as_ref(),
        )?
        .tasks
        .iter()
        .map(|task| task.row_group)
        .collect())
    }

    fn try_execute_direct_primitive_fold_metrics(
        &self,
        node: PhysicalPlanNode,
    ) -> Result<Option<AggregateMetrics>> {
        let Some(PhysicalExecutionConfig::DirectPrimitiveFold {
            path,
            batch_size,
            row_groups,
            columns,
            mode,
        }) = node.execution_config().cloned()
        else {
            return Err(DodamError::UnsupportedSql(
                "DirectPrimitiveFoldExec node is missing execution config".to_string(),
            ));
        };
        DirectPrimitiveFoldExec::new(
            path,
            batch_size,
            row_groups,
            columns,
            mode,
            self.file_cache.clone(),
            self.object_store.clone(),
        )
        .try_execute_metrics()
        .or_else(|error| match &error {
            DodamError::UnsupportedSql(message)
                if message.starts_with("unsupported DirectPrimitiveFoldExec ") =>
            {
                Ok(None)
            }
            _ => Err(error),
        })
    }

    pub(crate) fn parquet_decimal128_type(
        &self,
        path: impl AsRef<Path>,
        column: &str,
    ) -> Result<Option<(u8, i8)>> {
        let path = path.as_ref();
        let metadata = self
            .metadata_cache
            .get_with_store(path, self.object_store.as_ref())?;
        let Ok(field) = metadata.schema().field_with_name(column) else {
            return Ok(None);
        };
        match field.data_type() {
            DataType::Decimal128(precision, scale) => Ok(Some((*precision, *scale))),
            _ => Ok(None),
        }
    }

    pub(crate) fn parquet_is_date32_column(
        &self,
        path: impl AsRef<Path>,
        column: &str,
    ) -> Result<bool> {
        let path = path.as_ref();
        let metadata = self
            .metadata_cache
            .get_with_store(path, self.object_store.as_ref())?;
        let Ok(field) = metadata.schema().field_with_name(column) else {
            return Ok(false);
        };
        Ok(matches!(field.data_type(), DataType::Date32))
    }

    pub(crate) fn parquet_i128_column_min_max(
        &self,
        path: impl AsRef<Path>,
        column: &str,
    ) -> Result<Option<(i128, i128)>> {
        let path = path.as_ref();
        let object_metadata = self.object_store.metadata(path)?;
        let key = I128ColumnMinMaxCacheKey {
            path: path.to_path_buf(),
            len: object_metadata.len,
            modified_nanos: object_metadata
                .modified
                .and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
                .map(|duration| duration.as_nanos()),
            column: column.to_string(),
        };
        if let Some(value) = self
            .i128_column_min_max_cache
            .lock()
            .expect("i128 column min/max cache lock")
            .get(&key)
            .copied()
        {
            return Ok(value);
        }
        let value = read_parquet_i128_column_min_max(
            path,
            column,
            &self.metadata_cache,
            self.object_store.as_ref(),
        )?;
        self.i128_column_min_max_cache
            .lock()
            .expect("i128 column min/max cache lock")
            .insert(key, value);
        Ok(value)
    }

    pub(crate) fn parquet_i128_column_min_max_relaxed(
        &self,
        path: impl AsRef<Path>,
        column: &str,
    ) -> Result<Option<(i128, i128)>> {
        read_parquet_i128_column_min_max_relaxed(
            path,
            column,
            &self.metadata_cache,
            self.object_store.as_ref(),
        )
    }

    fn parquet_primitive_key_type(
        &self,
        path: impl AsRef<Path>,
        column: &str,
    ) -> Result<Option<DirectPrimitiveKeyType>> {
        let path = path.as_ref();
        let file = self.object_store.open(path)?;
        let metadata = self
            .metadata_cache
            .get_with_store(path, self.object_store.as_ref())?;
        let builder = ParquetRecordBatchReaderBuilder::new_with_metadata(file, metadata);
        let Ok(field) = builder.schema().field_with_name(column) else {
            return Ok(None);
        };
        match field.data_type() {
            DataType::Int32 => Ok(Some(DirectPrimitiveKeyType::I32)),
            DataType::Int64 => Ok(Some(DirectPrimitiveKeyType::I64)),
            DataType::Utf8 => Ok(Some(DirectPrimitiveKeyType::DictionaryI32Utf8)),
            DataType::Dictionary(_, value_type)
                if matches!(value_type.as_ref(), DataType::Utf8) =>
            {
                Ok(Some(DirectPrimitiveKeyType::DictionaryI32Utf8))
            }
            _ => Ok(None),
        }
    }

    async fn try_late_materialized_parquet_aggregate(
        &self,
        path: PathBuf,
        batch_size: usize,
        aggregates: Vec<AggregateExpr>,
        group_by: Vec<String>,
        filter: FilterExpr,
    ) -> Result<Option<AggregateMetrics>> {
        if !late_materialized_aggregate_enabled() {
            return Ok(None);
        }
        let column_read_plan = aggregate_column_read_plan(&aggregates, &group_by, Some(&filter));
        if projection_column_count(&column_read_plan.predicate_projection) == Some(0) {
            return Ok(None);
        }
        if matches!(column_read_plan.payload_projection, Projection::All)
            || projection_column_count(&column_read_plan.payload_projection) == Some(0)
        {
            return Ok(None);
        }
        log_aggregate_column_read_plan(&column_read_plan);
        let payload_dictionary_columns =
            late_materialized_aggregate_dictionary_columns(&aggregates, &group_by);
        let Some(partials) = self
            .late_materialized_parquet_map_pruned_with_policy_view_dictionary_columns(
                path,
                batch_size,
                column_read_plan.predicate_projection.clone(),
                column_read_plan.payload_projection.clone(),
                payload_dictionary_columns,
                Vec::new(),
                late_materialized_aggregate_row_group_chunk(),
                LateMaterializationPolicy::selective_with_selector_run_ratio(
                    late_materialized_aggregate_max_selected_ratio(),
                    late_materialized_aggregate_max_selector_run_ratio(),
                ),
                Vec::<RecordBatch>::new,
                {
                    let filter = filter.clone();
                    move |view, selection, _batches| {
                        let Some(batch) = view.try_record_batch() else {
                            return Err(DodamError::UnsupportedSql(
                                "late materialized aggregate selection requires RecordBatch"
                                    .to_string(),
                            ));
                        };
                        let mask = evaluate_filter_mask(&batch, &filter)?;
                        for row in 0..mask.len() {
                            selection.push(mask.is_valid(row) && mask.value(row));
                        }
                        Ok(Some(()))
                    }
                },
                |view, batches| {
                    let Some(batch) = view.try_record_batch() else {
                        return Err(DodamError::UnsupportedSql(
                            "late materialized aggregate payload requires RecordBatch".to_string(),
                        ));
                    };
                    batches.push(batch.clone());
                    Ok(Some(()))
                },
                {
                    let aggregates = aggregates.clone();
                    let group_by = group_by.clone();
                    move |batches, _metrics| {
                        let stream = Box::new(MemoryExec::new(batches)).execute()?;
                        let metrics = if group_by.is_empty() {
                            collect_aggregates(stream, 1, &aggregates)?
                        } else {
                            collect_grouped_aggregates(stream, 1, &group_by, &aggregates)?
                        };
                        Ok(Some(metrics))
                    }
                },
            )
            .await?
        else {
            return Ok(None);
        };
        let mut late_metrics = LateMaterializedMetrics::default();
        let mut partial_metrics = Vec::with_capacity(partials.len());
        for partial in partials {
            late_metrics.add(partial.metrics);
            partial_metrics.push(partial.output);
        }
        log_late_materialized_metrics("Aggregate", late_metrics, partial_metrics.len());
        if partial_metrics.is_empty() {
            return Ok(None);
        }
        merge_partial_aggregate_metrics(partial_metrics, 1, &group_by, &aggregates).map(Some)
    }

    pub fn plan_table_aggregate(
        &self,
        source: TableScanSource,
        batch_size: usize,
        aggregates: Vec<AggregateExpr>,
        group_by: Vec<String>,
        filter: Option<FilterExpr>,
    ) -> Result<AggregatePlan> {
        let column_read_plan = aggregate_column_read_plan(&aggregates, &group_by, filter.as_ref());
        let direct_physical =
            if fused_parquet_aggregate_enabled() && can_merge_partial_aggregates(&aggregates) {
                self.try_build_direct_primitive_fold_node(
                    &source,
                    batch_size,
                    &aggregates,
                    &group_by,
                    filter.as_ref(),
                )?
            } else {
                None
            };
        let plan = self.plan_table_scan(
            source,
            batch_size,
            None,
            column_read_plan.payload_projection.clone(),
            filter,
            None,
        )?;
        log_aggregate_column_read_plan(&column_read_plan);
        Ok(AggregatePlan {
            scan: plan,
            aggregates,
            group_by,
            column_read_plan,
            direct_physical,
        })
    }

    pub async fn plan_parquet_aggregate(
        &self,
        path: PathBuf,
        batch_size: usize,
        aggregates: Vec<AggregateExpr>,
        group_by: Vec<String>,
        filter: Option<FilterExpr>,
    ) -> Result<AggregatePlan> {
        let column_read_plan = aggregate_column_read_plan(&aggregates, &group_by, filter.as_ref());
        let source = self.plan_table_source(path).await?;
        let direct_physical =
            if fused_parquet_aggregate_enabled() && can_merge_partial_aggregates(&aggregates) {
                self.try_build_direct_primitive_fold_node(
                    &source,
                    batch_size,
                    &aggregates,
                    &group_by,
                    filter.as_ref(),
                )?
            } else {
                None
            };
        let scan = self.plan_table_scan(
            source,
            batch_size,
            None,
            column_read_plan.payload_projection.clone(),
            filter,
            None,
        )?;
        log_aggregate_column_read_plan(&column_read_plan);
        Ok(AggregatePlan {
            scan,
            aggregates,
            group_by,
            column_read_plan,
            direct_physical,
        })
    }

    pub async fn explain_parquet_aggregate(
        &self,
        path: PathBuf,
        batch_size: usize,
        aggregates: Vec<AggregateExpr>,
        group_by: Vec<String>,
        filter: Option<FilterExpr>,
    ) -> Result<String> {
        Ok(self
            .plan_parquet_aggregate(path, batch_size, aggregates, group_by, filter)
            .await?
            .explain())
    }

    pub fn explain_table_aggregate(
        &self,
        source: TableScanSource,
        batch_size: usize,
        aggregates: Vec<AggregateExpr>,
        group_by: Vec<String>,
        filter: Option<FilterExpr>,
    ) -> Result<String> {
        Ok(self
            .plan_table_aggregate(source, batch_size, aggregates, group_by, filter)?
            .explain())
    }

    fn execute_aggregate_plan(&self, plan: AggregatePlan) -> Result<AggregateMetrics> {
        let started = Instant::now();
        if let Some(node) = plan.direct_physical.clone()
            && let Some(mut metrics) = self.try_execute_direct_primitive_fold_metrics(node)?
        {
            metrics.aggregate_nanos = elapsed_nanos(started.elapsed());
            log_aggregate_profile(
                "aggregate_plan_direct_primitive",
                &metrics,
                started.elapsed(),
            );
            return Ok(metrics);
        }
        let fragment_count = plan.scan.source.fragments.len();
        let stream = self.execute_scan_plan(plan.scan)?;
        let metrics = if plan.group_by.is_empty() {
            collect_aggregates(stream, fragment_count, &plan.aggregates)
        } else {
            collect_grouped_aggregates(stream, fragment_count, &plan.group_by, &plan.aggregates)
        }?;
        log_aggregate_profile("aggregate_plan", &metrics, started.elapsed());
        Ok(metrics)
    }

    pub async fn plan_table_source(&self, path: PathBuf) -> Result<TableScanSource> {
        if let Some(source) = self.resolve_catalog_table_source(&path)? {
            return Ok(source);
        }
        let table_path = self.resolve_table_path(path)?;
        let table = LocalParquetTable::new(table_path);
        self.enrich_table_source(table.scan_source()?)
    }

    fn resolve_catalog_table_source(
        &self,
        path_or_table: &Path,
    ) -> Result<Option<TableScanSource>> {
        if path_or_table.exists() || path_or_table.components().count() != 1 {
            return Ok(None);
        }
        let Some(name) = path_or_table.to_str() else {
            return Ok(None);
        };
        PersistentCatalog::new(&self.catalog_root).table_scan_source(name)
    }

    fn resolve_table_path(&self, path_or_table: PathBuf) -> Result<PathBuf> {
        if path_or_table.exists() {
            return Ok(path_or_table);
        }
        if path_or_table.components().count() == 1 {
            let Some(name) = path_or_table.to_str() else {
                return Ok(path_or_table);
            };
            if let Some(entry) = PersistentCatalog::new(&self.catalog_root).table(name)? {
                return Ok(PathBuf::from(entry.location));
            }
        }
        Ok(path_or_table)
    }

    fn enrich_table_source(&self, mut source: TableScanSource) -> Result<TableScanSource> {
        if source.format != StorageFormat::Parquet {
            return Ok(source);
        }

        let mut schema = source.schema.clone();
        let mut fragments = Vec::with_capacity(source.fragments.len());
        for fragment in source.fragments {
            let statistics = read_parquet_file_statistics(
                fragment.parquet_local_path()?,
                &self.metadata_cache,
                self.object_store.as_ref(),
            )?;
            match &schema {
                Some(existing) if existing.as_ref() != statistics.schema.as_ref() => {
                    return Err(DodamError::UnsupportedSql(
                        "table fragments must have identical schemas".to_string(),
                    ));
                }
                Some(_) => {}
                None => schema = Some(statistics.schema.clone()),
            }
            fragments.push(fragment.with_statistics(FileFragmentStatistics {
                rows: statistics.rows,
                row_groups: statistics.row_groups,
                compressed_bytes: statistics.compressed_bytes,
            }));
        }

        source.schema = schema;
        source.fragments = fragments;
        source.statistics = TableStatistics::from_fragments(&source.fragments);
        Ok(source)
    }

    fn estimate_scan_source_bytes(
        &self,
        source: &TableScanSource,
        projection: &Projection,
        filter: Option<&FilterExpr>,
    ) -> Result<u64> {
        if source.format == StorageFormat::Parquet
            && matches!(projection, Projection::All)
            && filter.is_none()
            && source.statistics.fragments == source.fragments.len()
        {
            return Ok(source.statistics.compressed_bytes);
        }

        let predicates = PredicateSet::new(filter.cloned());
        let mut bytes = 0_u64;
        for fragment in &source.fragments {
            let parquet_projection =
                projection_without_partition_columns(projection, &fragment.partition_values);
            let plan = plan_parquet_scan_tasks(
                fragment.parquet_local_path()?,
                &parquet_projection,
                predicates.pushdown(),
                &self.metadata_cache,
                self.object_store.as_ref(),
            )?;
            bytes = bytes.saturating_add(plan.compressed_bytes_scanned);
        }
        Ok(bytes)
    }

    fn execute_scan_plan(&self, plan: ScanPlan) -> Result<SendableBatchStream> {
        let profile_label = scan_profile_label(&plan);
        let stream = self.build_physical_scan_plan(plan).execute()?;
        Ok(wrap_scan_profile(stream, profile_label))
    }

    fn build_physical_scan_plan(&self, plan: ScanPlan) -> Box<dyn PhysicalPlan> {
        let needs_output_projection = plan.scan_projection != plan.output_projection;
        let scan = ScanExec::new(
            plan.source.fragments,
            plan.batch_size,
            plan.scan_projection,
            plan.pushdown_predicates,
            plan.row_filter_predicates,
            self.metadata_cache.clone(),
            self.file_cache.clone(),
            self.object_store.clone(),
            plan.preserve_order,
        );
        let mut physical: Box<dyn PhysicalPlan> = Box::new(scan);

        if let Some(filter) = plan.residual_filter {
            physical = Box::new(FilterExec::new(physical, filter));
        }

        if plan.distinct {
            if needs_output_projection {
                physical = Box::new(ProjectionExec::new(
                    physical,
                    plan.output_projection.clone(),
                ));
            }
            physical = Box::new(DistinctExec::new(physical));

            if let Some(order_by) = plan.order_by {
                physical = Box::new(SortExec::new(physical, order_by, plan.limit));
            }
        } else {
            if let Some(order_by) = plan.order_by {
                physical = Box::new(SortExec::new(physical, order_by, plan.limit));
            }
            if needs_output_projection {
                physical = Box::new(ProjectionExec::new(physical, plan.output_projection));
            }
        }

        if let Some(limit) = plan.limit {
            physical = Box::new(LimitExec::new(physical, limit));
        }

        physical
    }
}

fn parquet_map_profile_label(path: &Path, projection: &Projection) -> String {
    let table = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("scan");
    let projection = match projection {
        Projection::All => "*".to_string(),
        Projection::Columns(columns) => columns.join(","),
    };
    format!("{table}[{projection}]")
}

fn scan_profile_enabled() -> bool {
    std::env::var("DODAM_SCAN_PROFILE")
        .or_else(|_| std::env::var("DODAM_TPCH_PROFILE"))
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn direct_primitive_profile_enabled() -> bool {
    std::env::var("DODAM_DIRECT_PRIMITIVE_PROFILE")
        .or_else(|_| std::env::var("DODAM_TPCH_PROFILE"))
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

pub(crate) fn direct_selection_fold_enabled() -> bool {
    if std::env::var("DODAM_DISABLE_DIRECT_SELECTION_FOLD")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
    {
        return false;
    }
    std::env::var("DODAM_DIRECT_SELECTION_FOLD")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(true)
}

fn decimal_date_selected_typed_agg_enabled() -> bool {
    std::env::var("DODAM_ENABLE_DECIMAL_DATE_SELECTED_TYPED_AGG")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn option_i128_to_i64(value: Option<i128>) -> Result<Option<i64>> {
    value
        .map(|value| {
            i64::try_from(value).map_err(|_| {
                DodamError::UnsupportedSql(
                    "direct selection decimal bound is outside Int64 range".to_string(),
                )
            })
        })
        .transpose()
}

fn direct_primitive_fold_row_groups(node: &PhysicalPlanNode) -> Result<Vec<usize>> {
    let Some(PhysicalExecutionConfig::DirectPrimitiveFold { row_groups, .. }) =
        node.execution_config()
    else {
        return Err(DodamError::UnsupportedSql(
            "DirectPrimitiveFoldExec node is missing execution config".to_string(),
        ));
    };
    Ok(row_groups.clone())
}

fn log_direct_primitive_fold_profile(
    path: &Path,
    columns: &[OwnedDirectPrimitiveColumnSpec],
    metrics: &DirectPrimitiveColumnScanMetrics,
) {
    let columns = columns
        .iter()
        .map(|column| column.name.as_str())
        .collect::<Vec<_>>();
    log_direct_primitive_named_profile(path, &columns, metrics);
}

fn log_direct_primitive_named_profile(
    path: &Path,
    columns: &[&str],
    metrics: &DirectPrimitiveColumnScanMetrics,
) {
    if !direct_primitive_profile_enabled() {
        return;
    }
    let table = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("scan");
    let columns = columns.join(",");
    let column_read = metrics
        .column_read_nanos
        .iter()
        .enumerate()
        .map(|(index, nanos)| format!("{index}:{:.3}", nanos_to_millis(*nanos)))
        .collect::<Vec<_>>()
        .join(",");
    let selected_ratio = if metrics.rows == 0 {
        0.0
    } else {
        metrics.selected_rows as f64 / metrics.rows as f64
    };
    eprintln!(
        "[dodam:direct-primitive-profile] {table}[{columns}]: reader_kind={} row_groups={} rows={} batches={} read={:.3} ms consume={:.3} ms column_read=[{}] selected_predicate={:.3} ms selected_payload={:.3} ms selected_dictionary={:.3} ms selected_rows={} selected_ratio={:.3} selected_runs={} selected_batches={} full_batches={} selected_skip_calls={} selected_skipped_rows={} selected_read_calls={} selected_read_rows={} dictionary_range_pages={} dictionary_block_pages={} dictionary_block_rows={} selected_page_skip_pages={} selected_page_skip_rows={}",
        metrics.reader_kind,
        metrics.row_groups,
        metrics.rows,
        metrics.batches,
        nanos_to_millis(metrics.read_nanos),
        nanos_to_millis(metrics.consume_nanos),
        column_read,
        nanos_to_millis(metrics.selected_predicate_nanos),
        nanos_to_millis(metrics.selected_payload_nanos),
        nanos_to_millis(metrics.selected_dictionary_nanos),
        metrics.selected_rows,
        selected_ratio,
        metrics.selected_runs,
        metrics.selected_payload_batches,
        metrics.full_payload_batches,
        metrics.selected_skip_calls,
        metrics.selected_skipped_rows,
        metrics.selected_read_calls,
        metrics.selected_read_rows,
        metrics.selected_dictionary_range_pages,
        metrics.selected_dictionary_block_pages,
        metrics.selected_dictionary_block_rows,
        metrics.selected_page_skip_pages,
        metrics.selected_page_skip_rows,
    );
}

fn log_direct_column_scan_profile(
    path: &Path,
    columns: &[&str],
    label: &str,
    metrics: &DirectColumnScanMetrics,
) {
    if !direct_primitive_profile_enabled() {
        return;
    }
    let table = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("scan");
    let columns = columns.join(",");
    eprintln!(
        "[dodam:direct-column-profile] kind={label} {table}[{columns}]: row_groups={} rows={} batches={} read={:.3} ms consume={:.3} ms dict_read={:.3} ms numeric_read={:.3} ms",
        metrics.row_groups,
        metrics.rows,
        metrics.batches,
        nanos_to_millis(metrics.read_nanos),
        nanos_to_millis(metrics.consume_nanos),
        nanos_to_millis(metrics.selected_predicate_nanos),
        nanos_to_millis(metrics.selected_payload_nanos),
    );
}

fn scan_profile_label(plan: &ScanPlan) -> String {
    let table = plan
        .source
        .fragments
        .first()
        .and_then(|fragment| fragment.parquet_local_path().ok())
        .and_then(|path| path.file_name())
        .and_then(|name| name.to_str())
        .unwrap_or("scan");
    let projection = match &plan.scan_projection {
        Projection::All => "*".to_string(),
        Projection::Columns(columns) => columns.join(","),
    };
    format!("{table}[{projection}]")
}

fn wrap_scan_profile(stream: SendableBatchStream, label: String) -> SendableBatchStream {
    if !scan_profile_enabled() {
        return stream;
    }
    let (inner, metrics) = stream.into_parts();
    SendableBatchStream::new(
        Box::new(ProfiledScanStream {
            inner,
            metrics: metrics.clone(),
            label,
            started: Instant::now(),
            rows: 0,
            batches: 0,
            next_wait_nanos: 0,
            logged: false,
        }),
        metrics,
    )
}

struct ProfiledScanStream {
    inner: Box<dyn Iterator<Item = Result<RecordBatch>> + Send>,
    metrics: Arc<ScanPlanMetricsCounter>,
    label: String,
    started: Instant,
    rows: usize,
    batches: usize,
    next_wait_nanos: u64,
    logged: bool,
}

impl ProfiledScanStream {
    fn log_once(&mut self) {
        if self.logged {
            return;
        }
        self.logged = true;
        let metrics = self.metrics.snapshot();
        let elapsed_ms = self.started.elapsed().as_secs_f64() * 1000.0;
        let next_wait_ms = nanos_to_millis(self.next_wait_nanos);
        let consumer_gap_ms = (elapsed_ms - next_wait_ms).max(0.0);
        eprintln!(
            "[dodam:scan-profile] {}: elapsed={:.3} ms next_wait={:.3} ms consumer_gap={:.3} ms rows={} batches={} row_groups={}/{} bytes={} metadata={:.3} ms planning={:.3} ms decode={:.3} ms parquet_next={:.3} ms parquet_next_avg={:.3} ms parquet_next_max={:.3} ms parquet_calls={} parquet_eof={} parquet_rows={} parquet_batches={} parquet_zero_batches={} avg_batch_rows={:.1} filter={:.3} ms projection={:.3} ms limit={:.3} ms",
            self.label,
            elapsed_ms,
            next_wait_ms,
            consumer_gap_ms,
            self.rows,
            self.batches,
            metrics.row_groups_scanned,
            metrics.row_groups_total,
            metrics.compressed_bytes_scanned,
            nanos_to_millis(metrics.metadata_nanos),
            nanos_to_millis(metrics.planning_nanos),
            nanos_to_millis(metrics.decode_nanos),
            nanos_to_millis(metrics.parquet_next_nanos),
            average_nanos_millis(metrics.parquet_next_nanos, metrics.parquet_next_calls),
            nanos_to_millis(metrics.parquet_max_next_nanos),
            metrics.parquet_next_calls,
            metrics.parquet_eof_calls,
            metrics.parquet_output_rows,
            metrics.parquet_output_batches,
            metrics.parquet_zero_row_batches,
            average_rows_per_batch(metrics.parquet_output_rows, metrics.parquet_output_batches),
            nanos_to_millis(metrics.filter_nanos),
            nanos_to_millis(metrics.projection_nanos),
            nanos_to_millis(metrics.limit_nanos),
        );
    }
}

impl Iterator for ProfiledScanStream {
    type Item = Result<RecordBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        let started = Instant::now();
        let next = self.inner.next();
        self.next_wait_nanos = self
            .next_wait_nanos
            .saturating_add(started.elapsed().as_nanos().min(u64::MAX as u128) as u64);
        match next {
            Some(Ok(batch)) => {
                self.rows = self.rows.saturating_add(batch.num_rows());
                self.batches = self.batches.saturating_add(1);
                Some(Ok(batch))
            }
            Some(Err(error)) => Some(Err(error)),
            None => {
                self.log_once();
                None
            }
        }
    }
}

impl Drop for ProfiledScanStream {
    fn drop(&mut self) {
        self.log_once();
    }
}

fn nanos_to_millis(nanos: u64) -> f64 {
    nanos as f64 / 1_000_000.0
}

fn average_rows_per_batch(rows: usize, batches: usize) -> f64 {
    if batches == 0 {
        0.0
    } else {
        rows as f64 / batches as f64
    }
}

fn average_nanos_millis(nanos: u64, count: usize) -> f64 {
    if count == 0 {
        0.0
    } else {
        nanos_to_millis(nanos) / count as f64
    }
}

#[derive(Debug, Clone)]
struct DirectCountSumShape {
    key_column: String,
    key_type: DirectPrimitiveKeyType,
    count_expr: AggregateExpr,
    sum_column: String,
}

impl DirectCountSumShape {
    fn try_new(
        aggregates: &[AggregateExpr],
        group_by: &[String],
        engine: &DodamEngine,
        path: &Path,
    ) -> Result<Option<Self>> {
        let [key_column] = group_by else {
            return Ok(None);
        };
        let [
            count_expr @ (AggregateExpr::CountStar | AggregateExpr::Count(_)),
            AggregateExpr::Sum(sum_column),
        ] = aggregates
        else {
            return Ok(None);
        };
        if sum_column == key_column {
            return Ok(None);
        }
        if let AggregateExpr::Count(count_column) = count_expr
            && count_column != key_column
            && count_column != sum_column
        {
            return Ok(None);
        }
        let Some(key_type) = engine.parquet_primitive_key_type(path, key_column)? else {
            return Ok(None);
        };
        Ok(Some(Self {
            key_column: key_column.clone(),
            key_type,
            count_expr: count_expr.clone(),
            sum_column: sum_column.clone(),
        }))
    }

    fn projection_columns(&self) -> Vec<String> {
        let mut columns = vec![self.key_column.clone()];
        if self.sum_column != self.key_column {
            columns.push(self.sum_column.clone());
        }
        columns
    }
}

#[derive(Debug, Clone)]
struct DirectCountSumMinMaxShape {
    key_column: String,
    key_type: DirectPrimitiveKeyType,
    second_key_column: Option<String>,
    sum_column: String,
    min_decimal_column: String,
    filter_date_column: String,
    max_kind: CountSumMinMaxMaxKind,
    decimal_precision: u8,
    decimal_scale: i8,
    filter: DecimalDateRangeFilter,
}

impl DirectCountSumMinMaxShape {
    fn try_new(
        aggregates: &[AggregateExpr],
        group_by: &[String],
        filter: &FilterExpr,
        engine: &DodamEngine,
        path: &Path,
    ) -> Result<Option<Self>> {
        let [key_column, rest @ ..] = group_by else {
            return Ok(None);
        };
        let second_key_column = match rest {
            [] => None,
            [second_key_column] if engine.parquet_is_date32_column(path, second_key_column)? => {
                Some(second_key_column.clone())
            }
            _ => return Ok(None),
        };
        let [
            AggregateExpr::CountStar,
            AggregateExpr::Sum(sum_column),
            AggregateExpr::Min(min_decimal_column),
            AggregateExpr::Max(max_date_column),
        ] = aggregates
        else {
            return Ok(None);
        };
        let Some((decimal_precision, decimal_scale)) =
            engine.parquet_decimal128_type(path, min_decimal_column)?
        else {
            return Ok(None);
        };
        let Some(key_type) = engine.parquet_primitive_key_type(path, key_column)? else {
            return Ok(None);
        };
        if second_key_column.is_some()
            && matches!(key_type, DirectPrimitiveKeyType::DictionaryI32Utf8)
        {
            return Ok(None);
        }
        let (filter_date_column, max_kind) =
            if engine.parquet_is_date32_column(path, max_date_column)? {
                (max_date_column.clone(), CountSumMinMaxMaxKind::Date32)
            } else if max_date_column == min_decimal_column
                && engine
                    .parquet_decimal128_type(path, max_date_column)?
                    .is_some()
            {
                let Some(date_column) = direct_count_sum_min_max_filter_date_column(
                    filter,
                    engine,
                    path,
                    &[key_column, sum_column, min_decimal_column],
                )?
                else {
                    return Ok(None);
                };
                (
                    date_column,
                    CountSumMinMaxMaxKind::Decimal128 {
                        precision: decimal_precision,
                        scale: decimal_scale,
                    },
                )
            } else {
                return Ok(None);
            };
        let Some(filter) = DecimalDateRangeFilter::try_new(
            filter.expr(),
            min_decimal_column,
            &filter_date_column,
            decimal_scale,
        )?
        else {
            return Ok(None);
        };
        Ok(Some(Self {
            key_column: key_column.clone(),
            key_type,
            second_key_column,
            sum_column: sum_column.clone(),
            min_decimal_column: min_decimal_column.clone(),
            filter_date_column,
            max_kind,
            decimal_precision,
            decimal_scale,
            filter,
        }))
    }

    fn projection_columns(&self) -> Vec<String> {
        let mut columns = Vec::with_capacity(5);
        let mut ordered = vec![&self.key_column];
        if let Some(second_key_column) = &self.second_key_column {
            ordered.push(second_key_column);
        }
        ordered.extend([
            &self.sum_column,
            &self.min_decimal_column,
            &self.filter_date_column,
        ]);
        for column in ordered {
            if !columns.iter().any(|existing| existing == column) {
                columns.push(column.clone());
            }
        }
        columns
    }

    fn is_single_key(&self) -> bool {
        self.second_key_column.is_none()
    }
}

fn direct_count_sum_min_max_filter_date_column(
    filter: &FilterExpr,
    engine: &DodamEngine,
    path: &Path,
    excluded: &[&String],
) -> Result<Option<String>> {
    for column in filter.referenced_columns() {
        if excluded
            .iter()
            .any(|excluded| excluded.as_str() == column.as_str())
        {
            continue;
        }
        if engine.parquet_is_date32_column(path, &column)? {
            return Ok(Some(column));
        }
    }
    Ok(None)
}

fn aggregate_projection(aggregates: &[AggregateExpr], group_by: &[String]) -> Projection {
    let mut columns = Vec::new();
    for aggregate in aggregates {
        if let Some(column) = aggregate.referenced_column()
            && !columns.iter().any(|existing| existing == column)
        {
            columns.push(column.to_string());
        }
    }
    for column in group_by {
        if !columns.iter().any(|existing| existing == column) {
            columns.push(column.clone());
        }
    }

    if columns.is_empty() {
        Projection::All
    } else {
        Projection::Columns(columns)
    }
}

fn aggregate_column_read_plan(
    aggregates: &[AggregateExpr],
    group_by: &[String],
    filter: Option<&FilterExpr>,
) -> AggregateColumnReadPlan {
    let payload_projection = aggregate_projection(aggregates, group_by);
    let predicate_columns = filter
        .map(FilterExpr::referenced_columns)
        .unwrap_or_default();
    let predicate_projection = Projection::Columns(predicate_columns);
    let scan_projection = scan_projection(&payload_projection, filter);
    AggregateColumnReadPlan {
        payload_projection,
        predicate_projection,
        scan_projection,
    }
}

fn projection_column_count(projection: &Projection) -> Option<usize> {
    match projection {
        Projection::All => None,
        Projection::Columns(columns) => Some(columns.len()),
    }
}

fn log_aggregate_column_read_plan(plan: &AggregateColumnReadPlan) {
    if !column_read_profile_enabled() {
        return;
    }
    eprintln!(
        "[dodam:column-read-plan] aggregate: payload={} predicate={} normal_scan={}",
        projection_display(&plan.payload_projection),
        projection_display(&plan.predicate_projection),
        projection_display(&plan.scan_projection),
    );
}

fn column_read_profile_enabled() -> bool {
    std::env::var("DODAM_COLUMN_READ_PROFILE")
        .or_else(|_| std::env::var("DODAM_PARQUET_COLUMN_PROFILE"))
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn fused_parquet_aggregate_enabled() -> bool {
    std::env::var("DODAM_DISABLE_FUSED_PARQUET_AGGREGATE")
        .map(|value| !matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(true)
}

fn fused_parquet_aggregate_row_group_chunk() -> usize {
    row_group_map_chunk_size("DODAM_FUSED_PARQUET_AGG_ROW_GROUP_CHUNK", 4)
}

fn late_materialized_aggregate_enabled() -> bool {
    std::env::var("DODAM_ENABLE_LATE_MATERIALIZED_AGGREGATE")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

fn filtered_dictionary_aggregate_enabled() -> bool {
    !std::env::var("DODAM_DISABLE_FILTERED_DICTIONARY_AGGREGATE")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn filtered_dictionary_aggregate_row_group_chunk() -> usize {
    row_group_map_chunk_size("DODAM_FILTERED_DICTIONARY_AGG_ROW_GROUP_CHUNK", 1)
}

fn row_group_map_chunk_size(query_env: &str, default_chunk: usize) -> usize {
    let requested_chunk = std::env::var("DODAM_ROW_GROUP_MAP_CHUNK")
        .or_else(|_| std::env::var(query_env))
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0);
    choose_row_group_map_chunk(RowGroupMapChunkCostInput {
        requested_chunk,
        default_chunk,
    })
}

fn filtered_count_sum_aggregate_enabled() -> bool {
    std::env::var("DODAM_ENABLE_FILTERED_COUNT_SUM_AGGREGATE")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

fn direct_i32_utf8_count_sum_aggregate_enabled() -> bool {
    std::env::var("DODAM_ENABLE_DIRECT_I32_UTF8_COUNT_SUM_AGGREGATE")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

fn late_materialized_aggregate_dictionary_columns(
    aggregates: &[AggregateExpr],
    group_by: &[String],
) -> Vec<String> {
    if group_by.len() == 1
        && matches!(
            aggregates,
            [
                AggregateExpr::CountStar | AggregateExpr::Count(_),
                AggregateExpr::Sum(_)
            ] | [
                AggregateExpr::CountStar,
                AggregateExpr::Sum(_),
                AggregateExpr::Min(_),
                AggregateExpr::Max(_)
            ]
        )
    {
        return group_by.to_vec();
    }
    Vec::new()
}

fn aggregate_profile_enabled() -> bool {
    std::env::var("DODAM_AGGREGATE_PROFILE")
        .or_else(|_| std::env::var("DODAM_GENERIC_PROFILE"))
        .or_else(|_| std::env::var("DODAM_TPCH_PROFILE"))
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn optimizer_trace_enabled() -> bool {
    std::env::var("DODAM_OPTIMIZER_TRACE")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn log_optimizer_rule_trace(rule: &str, decision: &str, reason: &str) {
    if optimizer_trace_enabled() {
        eprintln!("[dodam:optimizer] rule={rule} decision={decision} reason=\"{reason}\"");
    }
}

fn log_aggregate_profile(label: &str, metrics: &AggregateMetrics, total: Duration) {
    if !aggregate_profile_enabled() {
        return;
    }
    eprintln!(
        "[dodam:aggregate-profile] {label}: fragments={} batches={} rows={} groups={} values={} total={:.3} ms aggregate={:.3} ms merge={:.3} ms other={:.3} ms",
        metrics.fragments,
        metrics.batches,
        metrics.rows,
        metrics.groups.len(),
        metrics.values.len(),
        total.as_secs_f64() * 1000.0,
        nanos_to_millis(metrics.aggregate_nanos),
        nanos_to_millis(metrics.aggregate_merge_nanos),
        (total.as_nanos() as f64
            - (metrics.aggregate_nanos as f64 + metrics.aggregate_merge_nanos as f64))
            .max(0.0)
            / 1_000_000.0,
    );
}

fn late_materialized_aggregate_row_group_chunk() -> usize {
    std::env::var("DODAM_LATE_AGG_ROW_GROUP_CHUNK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(16)
}

fn late_materialized_aggregate_max_selected_ratio() -> f64 {
    std::env::var("DODAM_LATE_AGG_MAX_SELECTED_RATIO")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(0.60)
}

fn late_materialized_aggregate_max_selector_run_ratio() -> f64 {
    std::env::var("DODAM_LATE_AGG_MAX_SELECTOR_RUN_RATIO")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(0.02)
}

fn prune_table_source_partitions(
    mut source: TableScanSource,
    filter: Option<FilterExpr>,
) -> (TableScanSource, Option<FilterExpr>) {
    let Some(filter) = filter else {
        return (source, None);
    };
    let partition_columns = source
        .fragments
        .iter()
        .flat_map(|fragment| fragment.partition_values.keys().cloned())
        .collect::<HashSet<_>>();
    if partition_columns.is_empty() {
        return (source, Some(filter));
    }

    let mut residual = Vec::new();
    let mut partition_predicates = Vec::new();
    for conjunct in filter.conjuncts() {
        if partition_predicate_supported(&conjunct, &partition_columns) {
            partition_predicates.push(conjunct);
        } else {
            residual.push(conjunct);
        }
    }
    if partition_predicates.is_empty() {
        return (source, Some(filter));
    }

    source.fragments.retain(|fragment| {
        partition_predicates
            .iter()
            .all(|predicate| partition_predicate_matches(predicate, &fragment.partition_values))
    });
    source.statistics = TableStatistics::from_fragments(&source.fragments);
    (source, exprs_to_filter(residual))
}

fn partition_predicate_supported(expr: &Expr, partition_columns: &HashSet<String>) -> bool {
    match expr {
        Expr::Comparison(comparison) => {
            partition_columns.contains(&comparison.column)
                && comparison.op == ComparisonOp::Eq
                && !matches!(comparison.value, LiteralValue::Null)
        }
        Expr::InList {
            column,
            negated,
            values,
            has_null,
        } => partition_columns.contains(column) && !*negated && !*has_null && !values.is_empty(),
        _ => false,
    }
}

fn partition_predicate_matches(
    expr: &Expr,
    partition_values: &std::collections::BTreeMap<String, String>,
) -> bool {
    match expr {
        Expr::Comparison(comparison) => partition_values
            .get(&comparison.column)
            .is_some_and(|value| value == &literal_partition_value(&comparison.value)),
        Expr::InList {
            column,
            values,
            has_null,
            ..
        } => {
            !*has_null
                && partition_values.get(column).is_some_and(|value| {
                    values
                        .iter()
                        .any(|literal| value == &literal_partition_value(literal))
                })
        }
        Expr::Boolean(_)
        | Expr::ColumnComparison { .. }
        | Expr::Like { .. }
        | Expr::IsNull { .. }
        | Expr::Not(_)
        | Expr::And(_, _)
        | Expr::Or(_, _) => false,
    }
}

fn literal_partition_value(value: &LiteralValue) -> String {
    match value {
        LiteralValue::Null => "NULL".to_string(),
        LiteralValue::Boolean(value) => value.to_string(),
        LiteralValue::Utf8(value) => value.clone(),
        LiteralValue::Int64(value) => value.to_string(),
        LiteralValue::Float64(value) => value.to_string(),
    }
}

fn projection_without_partition_columns(
    projection: &Projection,
    partition_values: &std::collections::BTreeMap<String, String>,
) -> Projection {
    let Projection::Columns(columns) = projection else {
        return Projection::All;
    };
    Projection::Columns(
        columns
            .iter()
            .filter(|column| !partition_values.contains_key(*column))
            .cloned()
            .collect(),
    )
}

fn exprs_to_filter(exprs: Vec<Expr>) -> Option<FilterExpr> {
    let mut exprs = exprs.into_iter();
    let first = exprs.next()?;
    Some(FilterExpr::new(exprs.fold(first, |left, right| {
        Expr::And(Box::new(left), Box::new(right))
    })))
}

fn logical_scan_plan(
    source: &TableScanSource,
    batch_size: usize,
    limit: Option<usize>,
    projection: Projection,
    filter: Option<FilterExpr>,
    order_by: Option<SortKey>,
    distinct: bool,
) -> LogicalPlan {
    let mut plan = LogicalPlan::TableScan(LogicalScan {
        source: plan_table_source_from_scan_source(source),
        batch_size,
        projection: Projection::All,
        filter: None,
        order_by: None,
        limit: None,
        distinct: false,
    });
    if let Some(filter) = filter {
        plan = LogicalPlan::Filter {
            input: Box::new(plan),
            filter,
        };
    }
    if distinct {
        plan = LogicalPlan::Projection {
            input: Box::new(plan),
            projection,
        };
        plan = LogicalPlan::Distinct {
            input: Box::new(plan),
        };
        if let Some(order_by) = order_by {
            plan = LogicalPlan::Sort {
                input: Box::new(plan),
                order_by,
                limit: None,
            };
        }
    } else {
        if let Some(order_by) = order_by {
            plan = LogicalPlan::Sort {
                input: Box::new(plan),
                order_by,
                limit: None,
            };
        }
        plan = LogicalPlan::Projection {
            input: Box::new(plan),
            projection,
        };
    }
    if let Some(limit) = limit {
        plan = LogicalPlan::Limit {
            input: Box::new(plan),
            limit,
        };
    }
    plan
}

fn plan_table_source_from_scan_source(source: &TableScanSource) -> PlanTableSource {
    PlanTableSource {
        fragments: source.fragments.clone(),
        format: source.format,
        statistics: source.statistics,
    }
}

fn physical_plan_with_scan_fragments(
    plan: PhysicalPlanNode,
    fragments: Vec<crate::catalog::FileFragment>,
) -> Result<PhysicalPlanNode> {
    let mut replacements = 0_usize;
    let plan = replace_scan_fragments(plan, &fragments, &mut replacements);
    if replacements == 1 {
        Ok(plan)
    } else {
        Err(DodamError::UnsupportedSql(format!(
            "task execution expects exactly one scan node, found {replacements}"
        )))
    }
}

struct LocalShuffleStore {
    root: PathBuf,
    files: HashMap<(usize, usize), Vec<LocalShuffleFile>>,
    sequence: usize,
    file_target_bytes: u64,
}

#[derive(Debug, Clone)]
struct LocalShuffleFile {
    path: PathBuf,
    batches: usize,
    rows: usize,
    bytes: u64,
}

impl LocalShuffleStore {
    fn new() -> Result<Self> {
        Self::new_with_file_target_bytes(LOCAL_SHUFFLE_FILE_TARGET_BYTES)
    }

    fn new_with_file_target_bytes(file_target_bytes: u64) -> Result<Self> {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or_default();
        let root = std::env::temp_dir().join(format!(
            "dodam-local-shuffle-{}-{nanos}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root)?;
        Ok(Self {
            root,
            files: HashMap::new(),
            sequence: 0,
            file_target_bytes,
        })
    }

    fn write_partition(
        &mut self,
        stage_id: usize,
        partition: usize,
        batches: &[RecordBatch],
    ) -> Result<LocalShuffleWriteMetrics> {
        let non_empty_batches = batches
            .iter()
            .filter(|batch| batch.num_rows() > 0)
            .collect::<Vec<_>>();
        let mut metrics = LocalShuffleWriteMetrics {
            batches: non_empty_batches.len(),
            rows: non_empty_batches
                .iter()
                .map(|batch| batch.num_rows())
                .sum::<usize>(),
            ..LocalShuffleWriteMetrics::default()
        };
        if !non_empty_batches.is_empty() {
            let mut file_batches = Vec::new();
            let mut file_bytes = 0_u64;
            for batch in non_empty_batches {
                let batch_bytes = record_batch_memory_size(batch).max(1);
                if !file_batches.is_empty()
                    && file_bytes.saturating_add(batch_bytes) > self.file_target_bytes
                {
                    let write_metrics =
                        self.write_partition_file(stage_id, partition, &file_batches)?;
                    metrics.files = metrics.files.saturating_add(write_metrics.files);
                    metrics.bytes = metrics.bytes.saturating_add(write_metrics.bytes);
                    metrics.write_nanos = metrics
                        .write_nanos
                        .saturating_add(write_metrics.write_nanos);
                    file_batches.clear();
                    file_bytes = 0;
                }
                file_bytes = file_bytes.saturating_add(batch_bytes);
                file_batches.push(batch);
            }
            if !file_batches.is_empty() {
                let write_metrics =
                    self.write_partition_file(stage_id, partition, &file_batches)?;
                metrics.files = metrics.files.saturating_add(write_metrics.files);
                metrics.bytes = metrics.bytes.saturating_add(write_metrics.bytes);
                metrics.write_nanos = metrics
                    .write_nanos
                    .saturating_add(write_metrics.write_nanos);
            }
        }
        self.files.entry((stage_id, partition)).or_default();
        Ok(metrics)
    }

    fn write_partition_file(
        &mut self,
        stage_id: usize,
        partition: usize,
        batches: &[&RecordBatch],
    ) -> Result<LocalShuffleWriteMetrics> {
        let mut metrics = LocalShuffleWriteMetrics::default();
        if !batches.is_empty() {
            let path = self.root.join(format!(
                "stage-{stage_id}-partition-{partition}-{}.arrow",
                self.sequence
            ));
            self.sequence = self.sequence.saturating_add(1);
            let started = Instant::now();
            write_shuffle_ipc_batches(&path, batches)?;
            metrics.write_nanos = elapsed_nanos(started.elapsed());
            metrics.bytes = std::fs::metadata(&path)?.len();
            metrics.files = 1;
            self.files
                .entry((stage_id, partition))
                .or_default()
                .push(LocalShuffleFile {
                    path,
                    batches: batches.len(),
                    rows: batches.iter().map(|batch| batch.num_rows()).sum(),
                    bytes: metrics.bytes,
                });
        }
        Ok(metrics)
    }

    fn partition_files(&self, stage_id: usize, partition: usize) -> Result<Vec<PathBuf>> {
        Ok(self
            .partition_artifacts(stage_id, partition)?
            .iter()
            .map(|file| file.path.clone())
            .collect())
    }

    fn partition_read_metrics(
        &self,
        stage_id: usize,
        partition: usize,
    ) -> Result<LocalShuffleReadMetrics> {
        let files = self.partition_artifacts(stage_id, partition)?;
        Ok(LocalShuffleReadMetrics {
            files: files.len(),
            batches: files.iter().map(|file| file.batches).sum(),
            rows: files.iter().map(|file| file.rows).sum(),
            bytes: files.iter().map(|file| file.bytes).sum(),
        })
    }

    fn partition_artifacts(
        &self,
        stage_id: usize,
        partition: usize,
    ) -> Result<&[LocalShuffleFile]> {
        let Some(files) = self.files.get(&(stage_id, partition)) else {
            return Err(DodamError::UnsupportedSql(format!(
                "missing shuffle partition stage={stage_id} partition={partition}"
            )));
        };
        Ok(files)
    }
}

impl Drop for LocalShuffleStore {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

fn write_shuffle_ipc_batches(path: &Path, batches: &[&RecordBatch]) -> Result<()> {
    let Some(first_batch) = batches.first() else {
        return Ok(());
    };
    let mut file = File::create(path)?;
    let mut writer = IpcFileWriter::try_new(&mut file, first_batch.schema().as_ref())?;
    writer.write(first_batch)?;
    for batch in batches.iter().skip(1) {
        writer.write(batch)?;
    }
    writer.finish()?;
    Ok(())
}

fn physical_plan_with_shuffle_inputs(
    plan: PhysicalPlanNode,
    inputs: &[(usize, usize)],
    shuffle_store: &LocalShuffleStore,
) -> Result<(PhysicalPlanNode, LocalShuffleReadMetrics)> {
    let mut replacements = 0_usize;
    let mut read_metrics = LocalShuffleReadMetrics::default();
    let plan = replace_shuffle_inputs(
        plan,
        inputs,
        shuffle_store,
        &mut replacements,
        &mut read_metrics,
    )?;
    if replacements > 0 {
        Ok((plan, read_metrics))
    } else {
        Err(DodamError::UnsupportedSql(
            "task execution expected at least one exchange input".to_string(),
        ))
    }
}

fn replace_shuffle_inputs(
    plan: PhysicalPlanNode,
    inputs: &[(usize, usize)],
    shuffle_store: &LocalShuffleStore,
    replacements: &mut usize,
    read_metrics: &mut LocalShuffleReadMetrics,
) -> Result<PhysicalPlanNode> {
    if let PhysicalOperator::Exchange(kind) = plan.operator() {
        let matching_inputs = exchange_matching_inputs(&plan, inputs)?;
        if matching_inputs.is_empty() {
            return Err(DodamError::UnsupportedSql(
                "exchange input stage has no matching shuffle inputs".to_string(),
            ));
        }
        let mut files = Vec::new();
        for input in matching_inputs {
            let partition_metrics = shuffle_store.partition_read_metrics(input.0, input.1)?;
            read_metrics.files = read_metrics.files.saturating_add(partition_metrics.files);
            read_metrics.batches = read_metrics
                .batches
                .saturating_add(partition_metrics.batches);
            read_metrics.rows = read_metrics.rows.saturating_add(partition_metrics.rows);
            read_metrics.bytes = read_metrics.bytes.saturating_add(partition_metrics.bytes);
            files.extend(shuffle_store.partition_files(input.0, input.1)?);
        }
        *replacements += 1;
        return match kind {
            ExchangeKind::Gather | ExchangeKind::HashRepartition { .. } => {
                Ok(PhysicalPlanNode::ipc(files))
            }
            ExchangeKind::Broadcast => Err(DodamError::UnsupportedSql(
                "local shuffle execution does not support broadcast exchange yet".to_string(),
            )),
        };
    }

    let PhysicalPlanNode::Operator(mut node) = plan;
    node.children = node
        .children
        .into_iter()
        .map(|child| {
            replace_shuffle_inputs(child, inputs, shuffle_store, replacements, read_metrics)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(PhysicalPlanNode::Operator(node))
}

fn exchange_matching_inputs<'a>(
    exchange: &PhysicalPlanNode,
    inputs: &'a [(usize, usize)],
) -> Result<Vec<&'a (usize, usize)>> {
    let Some(input_stage) = physical_plan_attr(exchange, "input_stage") else {
        return Ok(inputs.iter().collect());
    };
    let input_stage = input_stage.parse::<usize>().map_err(|_| {
        DodamError::UnsupportedSql(format!("invalid exchange input_stage={input_stage}"))
    })?;
    Ok(inputs
        .iter()
        .filter(|(stage_id, _)| *stage_id == input_stage)
        .collect())
}

fn physical_plan_attr<'a>(plan: &'a PhysicalPlanNode, key: &str) -> Option<&'a str> {
    let PhysicalPlanNode::Operator(node) = plan;
    node.attributes
        .iter()
        .find_map(|(attribute_key, value)| (attribute_key == key).then_some(value.as_str()))
}

fn elapsed_nanos(elapsed: Duration) -> u64 {
    elapsed.as_nanos().min(u64::MAX as u128) as u64
}

fn record_batch_memory_size(batch: &RecordBatch) -> u64 {
    batch.get_array_memory_size().min(u64::MAX as usize) as u64
}

fn replace_scan_fragments(
    plan: PhysicalPlanNode,
    fragments: &[crate::catalog::FileFragment],
    replacements: &mut usize,
) -> PhysicalPlanNode {
    let PhysicalPlanNode::Operator(mut node) = plan;
    if let Some(PhysicalExecutionConfig::Scan {
        fragments: scan_fragments,
        ..
    }) = &mut node.execution
    {
        *scan_fragments = fragments.to_vec();
        *replacements += 1;
    }
    node.children = node
        .children
        .into_iter()
        .map(|child| replace_scan_fragments(child, fragments, replacements))
        .collect();
    PhysicalPlanNode::Operator(node)
}

fn repartition_batches_by_hash(
    batches: &[RecordBatch],
    keys: &[String],
    partitions: usize,
) -> Result<Vec<Vec<RecordBatch>>> {
    if partitions == 0 {
        return Err(DodamError::UnsupportedSql(
            "hash repartition requires at least one partition".to_string(),
        ));
    }
    let mut output = vec![Vec::new(); partitions];
    for batch in batches {
        if batch.num_rows() == 0 {
            continue;
        }
        let key_arrays = keys
            .iter()
            .map(|key| {
                let index = physical_batch_column_index(batch, key)?;
                Ok(batch.column(index).clone())
            })
            .collect::<Result<Vec<_>>>()?;
        let converter = RowConverter::new(
            key_arrays
                .iter()
                .map(|array| SortField::new(array.data_type().clone()))
                .collect(),
        )?;
        let rows = converter.convert_columns(&key_arrays)?;
        let mut indices = vec![Vec::<u32>::new(); partitions];
        for (row_index, row) in rows.iter().enumerate() {
            let partition = hash_physical_row(&row.owned()) % partitions;
            indices[partition].push(u32::try_from(row_index).map_err(|_| {
                DodamError::UnsupportedSql(
                    "hash shuffle currently supports up to u32::MAX rows per batch".to_string(),
                )
            })?);
        }
        for (partition, partition_indices) in indices.into_iter().enumerate() {
            if partition_indices.is_empty() {
                continue;
            }
            let indices = UInt32Array::from(partition_indices);
            output[partition].push(take_record_batch(batch, &indices)?);
        }
    }
    Ok(output)
}

fn physical_batch_column_index(batch: &RecordBatch, column: &str) -> Result<usize> {
    if let Ok(index) = batch.schema().index_of(column) {
        return Ok(index);
    }
    let suffix = column
        .rsplit_once('.')
        .map(|(_, suffix)| suffix)
        .unwrap_or(column);
    batch
        .schema()
        .fields()
        .iter()
        .position(|field| field.name() == suffix)
        .ok_or_else(|| {
            DodamError::UnsupportedSql(format!(
                "unknown shuffle key column {column}; available columns=[{}]",
                batch
                    .schema()
                    .fields()
                    .iter()
                    .map(|field| field.name().as_str())
                    .collect::<Vec<_>>()
                    .join(",")
            ))
        })
}

#[derive(Debug, Clone)]
struct OrderedGroupSumPartial {
    first: Option<OrderedRowGroupBoundary<i64, f64>>,
    last: Option<OrderedRowGroupBoundary<i64, f64>>,
    middle: HashMap<i64, f64>,
}

impl OrderedGroupSumPartial {
    fn new() -> Self {
        Self {
            first: None,
            last: None,
            middle: HashMap::new(),
        }
    }

    fn push_run(&mut self, key: i64, sum: f64, threshold: f64) {
        let boundary = OrderedRowGroupBoundary { key, state: sum };
        if self.first.is_none() {
            self.first = Some(boundary);
            return;
        }
        if let Some(last) = self.last.replace(boundary)
            && last.state > threshold
        {
            self.middle.insert(last.key, last.state);
        }
    }

    fn finish_current(&mut self, key: Option<i64>, sum: f64, threshold: f64) {
        if let Some(key) = key {
            self.push_run(key, sum, threshold);
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct LateMaterializedMetrics {
    pub total_rows: usize,
    pub selected_rows: usize,
    pub selector_runs: usize,
    pub predicate_read_nanos: u128,
    pub payload_read_nanos: u128,
    pub predicate_batches: usize,
    pub payload_batches: usize,
    pub payload_rows: usize,
}

impl LateMaterializedMetrics {
    pub fn add(&mut self, other: Self) {
        self.total_rows = self.total_rows.saturating_add(other.total_rows);
        self.selected_rows = self.selected_rows.saturating_add(other.selected_rows);
        self.selector_runs = self.selector_runs.saturating_add(other.selector_runs);
        self.predicate_read_nanos = self
            .predicate_read_nanos
            .saturating_add(other.predicate_read_nanos);
        self.payload_read_nanos = self
            .payload_read_nanos
            .saturating_add(other.payload_read_nanos);
        self.predicate_batches = self
            .predicate_batches
            .saturating_add(other.predicate_batches);
        self.payload_batches = self.payload_batches.saturating_add(other.payload_batches);
        self.payload_rows = self.payload_rows.saturating_add(other.payload_rows);
    }

    pub fn selected_ratio(&self) -> f64 {
        if self.total_rows == 0 {
            0.0
        } else {
            self.selected_rows as f64 / self.total_rows as f64
        }
    }
}

#[derive(Clone, Copy)]
pub struct LateMaterializationPolicy {
    max_selected_ratio: Option<f64>,
    max_selector_run_ratio: Option<f64>,
    max_selector_runs_per_selected: Option<f64>,
    io_cost_gate: bool,
}

impl LateMaterializationPolicy {
    pub fn always() -> Self {
        Self {
            max_selected_ratio: None,
            max_selector_run_ratio: None,
            max_selector_runs_per_selected: None,
            io_cost_gate: false,
        }
    }

    pub fn selective(max_selected_ratio: f64) -> Self {
        Self {
            max_selected_ratio: Some(max_selected_ratio.clamp(0.0, 1.0)),
            max_selector_run_ratio: None,
            max_selector_runs_per_selected: None,
            io_cost_gate: false,
        }
    }

    pub fn selective_with_selector_run_ratio(
        max_selected_ratio: f64,
        max_selector_run_ratio: f64,
    ) -> Self {
        Self {
            max_selected_ratio: Some(max_selected_ratio.clamp(0.0, 1.0)),
            max_selector_run_ratio: Some(max_selector_run_ratio.clamp(0.0, 1.0)),
            max_selector_runs_per_selected: None,
            io_cost_gate: false,
        }
    }

    pub fn with_selector_runs_per_selected(mut self, max_selector_runs_per_selected: f64) -> Self {
        self.max_selector_runs_per_selected = max_selector_runs_per_selected
            .is_finite()
            .then_some(max_selector_runs_per_selected.max(0.0));
        self
    }

    pub fn with_io_cost_gate(mut self, enabled: bool) -> Self {
        self.io_cost_gate = enabled;
        self
    }

    fn has_selectivity_gate(&self) -> bool {
        self.max_selected_ratio.is_some()
            || self.max_selector_run_ratio.is_some()
            || self.max_selector_runs_per_selected.is_some()
    }
}

fn late_materialization_policy_accepts_with_io(
    policy: LateMaterializationPolicy,
    metrics: &LateMaterializedMetrics,
    predicate_compressed_bytes: u64,
    payload_compressed_bytes: u64,
) -> bool {
    choose_late_materialization(LateMaterializationCostInput {
        total_rows: metrics.total_rows,
        selected_rows: metrics.selected_rows,
        selector_runs: metrics.selector_runs,
        max_selected_ratio: policy.max_selected_ratio,
        max_selector_run_ratio: policy.max_selector_run_ratio,
        max_selector_runs_per_selected: policy.max_selector_runs_per_selected,
        io_cost_gate: policy.io_cost_gate,
        predicate_compressed_bytes,
        payload_compressed_bytes,
        min_io_saving_ratio: late_materialization_min_estimated_io_saving_ratio(),
        io_override: late_materialization_io_override_enabled(),
    })
}

fn log_late_materialization_policy_decision(
    label: &str,
    accepted: bool,
    metrics: &LateMaterializedMetrics,
    predicate_compressed_bytes: u64,
    payload_compressed_bytes: u64,
    policy: LateMaterializationPolicy,
) {
    if !std::env::var("DODAM_LATE_MATERIALIZATION_PROFILE")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        && !std::env::var("DODAM_COALESCE_AGG_PROFILE")
            .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
    {
        return;
    }
    let selected_ratio = if metrics.total_rows == 0 {
        0.0
    } else {
        metrics.selected_rows as f64 / metrics.total_rows as f64
    };
    let selector_run_ratio = if metrics.total_rows == 0 {
        0.0
    } else {
        metrics.selector_runs as f64 / metrics.total_rows as f64
    };
    let selector_runs_per_selected = if metrics.selected_rows == 0 {
        0.0
    } else {
        metrics.selector_runs as f64 / metrics.selected_rows as f64
    };
    let io_saving = crate::cost::late_materialization_estimated_io_saving(
        selected_ratio,
        predicate_compressed_bytes,
        payload_compressed_bytes,
    );
    eprintln!(
        "[dodam:late-materialization-profile] {label} accepted={} rows={} selected={} selected_ratio={:.6} selector_runs={} selector_run_ratio={:.6} selector_runs_per_selected={:.6} predicate_bytes={} payload_bytes={} io_saving={:.6} max_selected_ratio={} max_selector_run_ratio={} max_selector_runs_per_selected={} io_cost_gate={}",
        accepted,
        metrics.total_rows,
        metrics.selected_rows,
        selected_ratio,
        metrics.selector_runs,
        selector_run_ratio,
        selector_runs_per_selected,
        predicate_compressed_bytes,
        payload_compressed_bytes,
        io_saving,
        policy
            .max_selected_ratio
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| "n/a".to_string()),
        policy
            .max_selector_run_ratio
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| "n/a".to_string()),
        policy
            .max_selector_runs_per_selected
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| "n/a".to_string()),
        policy.io_cost_gate,
    );
}

fn late_materialization_min_estimated_io_saving_ratio() -> f64 {
    std::env::var("DODAM_LATE_MIN_ESTIMATED_IO_SAVING_RATIO")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(0.25)
        .clamp(0.0, 1.0)
}

fn late_materialization_io_override_enabled() -> bool {
    !std::env::var("DODAM_DISABLE_LATE_IO_OVERRIDE")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn late_materialized_direct_raw_i64_payload_enabled() -> bool {
    std::env::var("DODAM_ENABLE_LATE_DIRECT_RAW_I64_PAYLOAD")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn late_materialized_decimal_i64_payload_columns(
    path: &Path,
    payload_projection: &Projection,
    metadata_cache: &ParquetMetadataCache,
    object_store: &dyn ObjectStore,
) -> Result<Option<([String; 2], [DirectPrimitiveColumnType; 2])>> {
    let Projection::Columns(columns) = payload_projection else {
        return Ok(None);
    };
    let [first, second] = columns.as_slice() else {
        return Ok(None);
    };
    let metadata = metadata_cache.get_with_store(path, object_store)?;
    let Ok(first_field) = metadata.schema().field_with_name(first) else {
        return Ok(None);
    };
    let Ok(second_field) = metadata.schema().field_with_name(second) else {
        return Ok(None);
    };
    let DataType::Decimal128(first_precision, first_scale) = first_field.data_type() else {
        return Ok(None);
    };
    let DataType::Decimal128(second_precision, second_scale) = second_field.data_type() else {
        return Ok(None);
    };
    Ok(Some((
        [first.clone(), second.clone()],
        [
            DirectPrimitiveColumnType::Decimal128Int64Raw {
                precision: *first_precision,
                scale: *first_scale,
            },
            DirectPrimitiveColumnType::Decimal128Int64Raw {
                precision: *second_precision,
                scale: *second_scale,
            },
        ],
    )))
}

fn late_materialized_selector_runs_by_row_group(
    path: &Path,
    row_groups: &[usize],
    selectors: &[RowSelector],
    metadata_cache: &ParquetMetadataCache,
    object_store: &dyn ObjectStore,
) -> Result<Option<Vec<Vec<(usize, usize)>>>> {
    let metadata = metadata_cache.get_with_store(path, object_store)?;
    let mut row_group_rows = Vec::with_capacity(row_groups.len());
    for &row_group in row_groups {
        let Some(metadata) = metadata.metadata().row_groups().get(row_group) else {
            return Ok(None);
        };
        row_group_rows.push(usize::try_from(metadata.num_rows()).map_err(|_| {
            DodamError::UnsupportedSql("row group row count out of range".to_string())
        })?);
    }
    Ok(Some(split_late_selectors_by_row_group(
        selectors,
        &row_group_rows,
    )))
}

fn split_late_selectors_by_row_group(
    selectors: &[RowSelector],
    row_group_rows: &[usize],
) -> Vec<Vec<(usize, usize)>> {
    let mut runs = row_group_rows
        .iter()
        .map(|_| Vec::<(usize, usize)>::new())
        .collect::<Vec<_>>();
    let mut selector_start = 0usize;
    let mut row_group_index = 0usize;
    let mut row_group_start = 0usize;
    let mut row_group_end = row_group_rows.first().copied().unwrap_or(0);
    for selector in selectors {
        let selector_end = selector_start.saturating_add(selector.row_count);
        if selector.skip {
            selector_start = selector_end;
            continue;
        }
        while row_group_index < row_group_rows.len() && row_group_end <= selector_start {
            row_group_index += 1;
            row_group_start = row_group_end;
            row_group_end = row_group_end.saturating_add(
                row_group_rows
                    .get(row_group_index)
                    .copied()
                    .unwrap_or_default(),
            );
        }
        let mut start = selector_start;
        while row_group_index < row_group_rows.len() && start < selector_end {
            let end = selector_end.min(row_group_end);
            if end > start {
                runs[row_group_index].push((start - row_group_start, end - start));
            }
            start = end;
            if start >= row_group_end {
                row_group_index += 1;
                row_group_start = row_group_end;
                row_group_end = row_group_end.saturating_add(
                    row_group_rows
                        .get(row_group_index)
                        .copied()
                        .unwrap_or_default(),
                );
            }
        }
        selector_start = selector_end;
    }
    runs
}

pub struct LateMaterializedChunkResult<T> {
    pub output: T,
    pub metrics: LateMaterializedMetrics,
}

#[derive(Default)]
pub struct LateSelectionBuilder {
    selectors: Vec<RowSelector>,
    current_selected: Option<bool>,
    run_len: usize,
    total_rows: usize,
    selected_rows: usize,
}

impl LateSelectionBuilder {
    pub fn push(&mut self, selected: bool) {
        self.total_rows += 1;
        if selected {
            self.selected_rows += 1;
        }
        push_late_selector_run(
            &mut self.selectors,
            &mut self.current_selected,
            &mut self.run_len,
            selected,
        );
    }

    pub fn push_repeated(&mut self, row_count: usize, selected: bool) {
        if row_count == 0 {
            return;
        }
        append_late_selector_run(
            &mut self.selectors,
            &mut self.current_selected,
            &mut self.run_len,
            selected,
            row_count,
        );
        self.total_rows += row_count;
        if selected {
            self.selected_rows += row_count;
        }
    }

    pub fn push_selected_offsets<I>(&mut self, row_count: usize, selected_offsets: I)
    where
        I: IntoIterator<Item = usize>,
    {
        let mut consumed = 0usize;
        let mut appended_selected = 0usize;
        for selected_offset in selected_offsets {
            if selected_offset >= row_count || selected_offset < consumed {
                continue;
            }
            let skipped = selected_offset - consumed;
            if skipped > 0 {
                append_late_selector_run(
                    &mut self.selectors,
                    &mut self.current_selected,
                    &mut self.run_len,
                    false,
                    skipped,
                );
            }
            append_late_selector_run(
                &mut self.selectors,
                &mut self.current_selected,
                &mut self.run_len,
                true,
                1,
            );
            consumed = selected_offset + 1;
            appended_selected += 1;
        }
        if consumed < row_count {
            append_late_selector_run(
                &mut self.selectors,
                &mut self.current_selected,
                &mut self.run_len,
                false,
                row_count - consumed,
            );
        }
        self.total_rows += row_count;
        self.selected_rows += appended_selected;
    }

    pub fn push_selected_u32_offsets(&mut self, row_count: usize, selected_offsets: &[u32]) {
        let mut consumed = 0usize;
        let mut appended_selected = 0usize;
        for &selected_offset in selected_offsets {
            let selected_offset = selected_offset as usize;
            if selected_offset >= row_count || selected_offset < consumed {
                continue;
            }
            let skipped = selected_offset - consumed;
            if skipped > 0 {
                append_late_selector_run(
                    &mut self.selectors,
                    &mut self.current_selected,
                    &mut self.run_len,
                    false,
                    skipped,
                );
            }
            append_late_selector_run(
                &mut self.selectors,
                &mut self.current_selected,
                &mut self.run_len,
                true,
                1,
            );
            consumed = selected_offset + 1;
            appended_selected += 1;
        }
        if consumed < row_count {
            append_late_selector_run(
                &mut self.selectors,
                &mut self.current_selected,
                &mut self.run_len,
                false,
                row_count - consumed,
            );
        }
        self.total_rows += row_count;
        self.selected_rows += appended_selected;
    }

    pub fn push_selected_u32_offsets_coalesced<F>(
        &mut self,
        row_count: usize,
        selected_offsets: &[u32],
        max_gap: usize,
        mut push_payload_marker: F,
    ) where
        F: FnMut(Option<usize>),
    {
        let max_gap = choose_late_coalesce_max_gap(
            selected_offsets,
            max_gap,
            late_coalesce_selector_run_cost_rows(),
            late_coalesce_max_payload_expansion_rows_per_selected(),
        );
        let mut consumed = 0usize;
        let mut payload_rows = 0usize;
        for &selected_offset in selected_offsets {
            let selected_offset = selected_offset as usize;
            if selected_offset >= row_count || selected_offset < consumed {
                continue;
            }
            let skipped = selected_offset - consumed;
            if skipped > 0 {
                if max_gap > 0 && skipped <= max_gap && payload_rows > 0 {
                    append_late_selector_run(
                        &mut self.selectors,
                        &mut self.current_selected,
                        &mut self.run_len,
                        true,
                        skipped,
                    );
                    for _ in 0..skipped {
                        push_payload_marker(None);
                    }
                    payload_rows += skipped;
                } else {
                    append_late_selector_run(
                        &mut self.selectors,
                        &mut self.current_selected,
                        &mut self.run_len,
                        false,
                        skipped,
                    );
                }
            }
            append_late_selector_run(
                &mut self.selectors,
                &mut self.current_selected,
                &mut self.run_len,
                true,
                1,
            );
            push_payload_marker(Some(selected_offset));
            consumed = selected_offset + 1;
            payload_rows += 1;
        }
        if consumed < row_count {
            append_late_selector_run(
                &mut self.selectors,
                &mut self.current_selected,
                &mut self.run_len,
                false,
                row_count - consumed,
            );
        }
        self.total_rows += row_count;
        self.selected_rows += payload_rows;
    }

    fn finish(mut self) -> (Option<RowSelection>, LateMaterializedMetrics) {
        finish_late_selector_run(&mut self.selectors, self.current_selected, self.run_len);
        let metrics = LateMaterializedMetrics {
            total_rows: self.total_rows,
            selected_rows: self.selected_rows,
            selector_runs: self.selectors.len(),
            ..Default::default()
        };
        let selection = (self.selected_rows > 0).then(|| RowSelection::from(self.selectors));
        (selection, metrics)
    }

    fn finish_with_selectors(
        mut self,
    ) -> (
        Option<RowSelection>,
        Vec<RowSelector>,
        LateMaterializedMetrics,
    ) {
        finish_late_selector_run(&mut self.selectors, self.current_selected, self.run_len);
        let metrics = LateMaterializedMetrics {
            total_rows: self.total_rows,
            selected_rows: self.selected_rows,
            selector_runs: self.selectors.len(),
            ..Default::default()
        };
        let selectors = self.selectors;
        let selection = (self.selected_rows > 0).then(|| RowSelection::from(selectors.clone()));
        (selection, selectors, metrics)
    }
}

fn ordered_group_sum_row_group_chunk() -> usize {
    std::env::var("DODAM_ORDERED_GROUP_ROW_GROUP_CHUNK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(4)
}

fn parallel_i64_set_filter_row_group_chunk() -> usize {
    std::env::var("DODAM_I64_SET_FILTER_ROW_GROUP_CHUNK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(4)
}

fn selected_discount_revenue_row_group_chunk() -> usize {
    std::env::var("DODAM_SELECTED_DISCOUNT_REVENUE_ROW_GROUP_CHUNK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(2)
}

fn bool_lookup_discounted_revenue_row_group_chunk() -> usize {
    std::env::var("DODAM_BOOL_LOOKUP_DISCOUNTED_REVENUE_ROW_GROUP_CHUNK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(4)
}

fn late_materialization_sample_row_groups() -> usize {
    std::env::var("DODAM_LATE_SAMPLE_ROW_GROUPS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1)
}

fn late_coalesce_selector_run_cost_rows() -> usize {
    std::env::var("DODAM_LATE_COALESCE_SELECTOR_RUN_COST_ROWS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(16)
}

fn late_coalesce_max_payload_expansion_rows_per_selected() -> usize {
    std::env::var("DODAM_LATE_COALESCE_MAX_PAYLOAD_EXPANSION_PER_SELECTED")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(8)
}

fn late_materialization_sample_enabled(
    policy: LateMaterializationPolicy,
    predicate_columns: usize,
    payload_columns: usize,
) -> bool {
    if !policy.has_selectivity_gate() || late_materialization_sample_row_groups() == 0 {
        return false;
    }
    if std::env::var_os("DODAM_LATE_SAMPLE_FORCE").is_some() {
        return true;
    }
    policy.io_cost_gate || payload_columns >= predicate_columns.saturating_mul(2).max(1)
}

#[allow(clippy::too_many_arguments)]
fn sample_late_materialized_selection_view<State, BuildSelection>(
    path: PathBuf,
    batch_size: usize,
    row_groups: Vec<usize>,
    predicate_projection: &Projection,
    metadata_cache: &ParquetMetadataCache,
    file_cache: Arc<ParquetFileCache>,
    object_store: &dyn ObjectStore,
    mut state: State,
    mut build_selection: BuildSelection,
) -> Result<Option<LateMaterializedMetrics>>
where
    BuildSelection:
        for<'a> FnMut(BatchView<'a>, &mut LateSelectionBuilder, &mut State) -> Result<Option<()>>,
{
    let mut predicate_reader = ParquetBatchReader::try_new_with_row_groups(
        path,
        batch_size,
        predicate_projection,
        row_groups,
        metadata_cache,
        file_cache,
        object_store,
    )?;
    let mut selection_builder = LateSelectionBuilder::default();
    let mut predicate_read_nanos = 0_u128;
    let mut predicate_batches = 0_usize;
    loop {
        let read_started = Instant::now();
        let batch = predicate_reader.next();
        predicate_read_nanos =
            predicate_read_nanos.saturating_add(read_started.elapsed().as_nanos());
        let Some(batch) = batch else {
            break;
        };
        let batch = batch?;
        predicate_batches = predicate_batches.saturating_add(1);
        if build_selection(BatchView::new(&batch), &mut selection_builder, &mut state)?.is_none() {
            return Ok(None);
        }
    }
    let (_, mut metrics) = selection_builder.finish();
    metrics.predicate_read_nanos = predicate_read_nanos;
    metrics.predicate_batches = predicate_batches;
    Ok(Some(metrics))
}

#[allow(clippy::too_many_arguments)]
fn late_materialized_chunk_view<State, Output, BuildSelection, ConsumePayload, Finish>(
    path: PathBuf,
    batch_size: usize,
    row_groups: Vec<usize>,
    predicate_projection: &Projection,
    payload_projection: &Projection,
    payload_dictionary_columns: &[String],
    metadata_cache: &ParquetMetadataCache,
    file_cache: Arc<ParquetFileCache>,
    object_store: &dyn ObjectStore,
    policy: LateMaterializationPolicy,
    predicate_compressed_bytes: u64,
    payload_compressed_bytes: u64,
    mut state: State,
    mut build_selection: BuildSelection,
    mut consume_payload: ConsumePayload,
    finish: Finish,
) -> Result<Option<LateMaterializedChunkResult<Output>>>
where
    BuildSelection:
        for<'a> FnMut(BatchView<'a>, &mut LateSelectionBuilder, &mut State) -> Result<Option<()>>,
    ConsumePayload: for<'a> FnMut(BatchView<'a>, &mut State) -> Result<Option<()>>,
    Finish: FnOnce(State, LateMaterializedMetrics) -> Result<Option<Output>>,
{
    let mut predicate_reader = ParquetBatchReader::try_new_with_row_groups(
        path.clone(),
        batch_size,
        predicate_projection,
        row_groups.clone(),
        metadata_cache,
        file_cache.clone(),
        object_store,
    )?;
    let mut selection_builder = LateSelectionBuilder::default();
    let mut predicate_read_nanos = 0_u128;
    let mut predicate_batches = 0_usize;
    loop {
        let read_started = Instant::now();
        let batch = predicate_reader.next();
        predicate_read_nanos =
            predicate_read_nanos.saturating_add(read_started.elapsed().as_nanos());
        let Some(batch) = batch else {
            break;
        };
        let batch = batch?;
        predicate_batches = predicate_batches.saturating_add(1);
        if build_selection(BatchView::new(&batch), &mut selection_builder, &mut state)?.is_none() {
            return Ok(None);
        }
    }
    let (row_selection, selectors, mut metrics) = selection_builder.finish_with_selectors();
    metrics.predicate_read_nanos = predicate_read_nanos;
    metrics.predicate_batches = predicate_batches;
    if !late_materialization_policy_accepts_with_io(
        policy,
        &metrics,
        predicate_compressed_bytes,
        payload_compressed_bytes,
    ) {
        return Ok(None);
    }
    if let Some(row_selection) = row_selection {
        if payload_dictionary_columns.is_empty()
            && late_materialized_direct_raw_i64_payload_enabled()
            && let Some((payload_columns, payload_types)) =
                late_materialized_decimal_i64_payload_columns(
                    &path,
                    payload_projection,
                    metadata_cache,
                    object_store,
                )?
            && let Some(row_group_runs) = late_materialized_selector_runs_by_row_group(
                &path,
                &row_groups,
                &selectors,
                metadata_cache,
                object_store,
            )?
        {
            if let Some(direct_metrics) = scan_parquet_i64_i64_selected_raw_columns_with_store(
                &path,
                &row_groups,
                [payload_columns[0].as_str(), payload_columns[1].as_str()],
                payload_types,
                &row_group_runs,
                file_cache.clone(),
                object_store,
                |columns| {
                    consume_payload(BatchView::from_raw_columns(columns), &mut state)?.ok_or_else(
                        || {
                            DodamError::UnsupportedSql(
                                "direct late raw payload consumer rejected batch".to_string(),
                            )
                        },
                    )
                },
            )? {
                metrics.payload_read_nanos = u128::from(direct_metrics.read_nanos);
                metrics.payload_batches = direct_metrics.batches;
                metrics.payload_rows = direct_metrics.selected_rows;
                let output = finish(state, metrics)?;
                return Ok(output.map(|output| LateMaterializedChunkResult { output, metrics }));
            }
        }
        let mut payload_reader = if payload_dictionary_columns.is_empty() {
            ParquetBatchReader::try_new_with_row_groups_selection(
                path,
                batch_size,
                payload_projection,
                row_groups,
                row_selection,
                metadata_cache,
                file_cache,
                object_store,
            )?
        } else {
            ParquetBatchReader::try_new_with_row_groups_selection_dictionary_columns(
                path,
                batch_size,
                payload_projection,
                row_groups,
                row_selection,
                payload_dictionary_columns,
                metadata_cache,
                file_cache,
                object_store,
            )?
        };
        loop {
            let read_started = Instant::now();
            let batch = payload_reader.next();
            metrics.payload_read_nanos = metrics
                .payload_read_nanos
                .saturating_add(read_started.elapsed().as_nanos());
            let Some(batch) = batch else {
                break;
            };
            let batch = batch?;
            metrics.payload_batches = metrics.payload_batches.saturating_add(1);
            metrics.payload_rows = metrics.payload_rows.saturating_add(batch.num_rows());
            if consume_payload(BatchView::new(&batch), &mut state)?.is_none() {
                return Ok(None);
            }
        }
    }
    let Some(output) = finish(state, metrics)? else {
        return Ok(None);
    };
    Ok(Some(LateMaterializedChunkResult { output, metrics }))
}

#[allow(clippy::too_many_arguments)]
fn parquet_row_group_map_chunk<State, Output, ConsumeBatch, Finish>(
    path: PathBuf,
    batch_size: usize,
    row_groups: Vec<usize>,
    projection: &Projection,
    metadata_cache: &ParquetMetadataCache,
    file_cache: Arc<ParquetFileCache>,
    object_store: &dyn ObjectStore,
    mut state: State,
    mut consume_batch: ConsumeBatch,
    finish: Finish,
    profile_label: Option<&str>,
    profile_chunk: usize,
) -> Result<Option<ParquetMapChunkResult<Output>>>
where
    ConsumeBatch: FnMut(RecordBatch, &mut State) -> Result<Option<()>>,
    Finish: FnOnce(State) -> Result<Option<Output>>,
{
    let started = profile_label.map(|_| Instant::now());
    let mut reader = ParquetBatchReader::try_new_with_row_groups(
        path,
        batch_size,
        projection,
        row_groups.clone(),
        metadata_cache,
        file_cache,
        object_store,
    )?;
    let reader_setup_nanos = started
        .map(|started| elapsed_nanos(started.elapsed()))
        .unwrap_or_default();
    let mut batches = 0_usize;
    let mut rows = 0_usize;
    let mut read_nanos = 0_u64;
    let mut consume_nanos = 0_u64;
    loop {
        let read_start = Instant::now();
        let Some(batch) = reader.next() else {
            read_nanos = read_nanos.saturating_add(elapsed_nanos(read_start.elapsed()));
            break;
        };
        let batch = batch?;
        read_nanos = read_nanos.saturating_add(elapsed_nanos(read_start.elapsed()));
        batches += 1;
        rows += batch.num_rows();
        let consume_start = Instant::now();
        if consume_batch(batch, &mut state)?.is_none() {
            return Ok(None);
        }
        consume_nanos = consume_nanos.saturating_add(elapsed_nanos(consume_start.elapsed()));
    }
    let total_nanos = started
        .map(|started| elapsed_nanos(started.elapsed()))
        .unwrap_or_default();
    let metrics = ParquetMapChunkMetrics {
        chunks: 1,
        row_groups: row_groups.len(),
        projected_columns: reader.projected_columns(),
        rows,
        batches,
        zero_batches: reader.zero_row_batches(),
        total_nanos,
        read_nanos,
        reader_next_nanos: reader.next_nanos(),
        reader_p95_next_nanos: reader.p95_next_nanos(),
        reader_max_next_nanos: reader.max_next_nanos(),
        reader_calls: reader.next_calls(),
        reader_eof: reader.eof_calls(),
        consume_nanos,
        compressed_bytes_scanned: reader.compressed_bytes_scanned(),
        compressed_bytes_total: reader.compressed_bytes_total(),
    };
    let finished = finish(state);
    if let (Some(label), Some(_)) = (profile_label, started) {
        eprintln!(
            "[dodam:tpch-profile] row_group_map {label}: chunk={} row_groups={} projected_columns={} rows={} batches={} zero_batches={} total={:.3} ms setup={:.3} ms metadata={:.3} ms planning={:.3} ms read_next={:.3} ms reader_next={:.3} ms reader_next_avg={:.3} ms reader_next_p95={:.3} ms reader_next_max={:.3} ms reader_calls={} reader_eof={} avg_batch_rows={:.1} consume={:.3} ms compressed={}/{}",
            profile_chunk,
            row_groups.len(),
            reader.projected_columns(),
            rows,
            batches,
            reader.zero_row_batches(),
            nanos_to_millis(metrics.total_nanos),
            nanos_to_millis(reader_setup_nanos),
            nanos_to_millis(reader.metadata_nanos()),
            nanos_to_millis(reader.planning_nanos()),
            nanos_to_millis(read_nanos),
            nanos_to_millis(reader.next_nanos()),
            average_nanos_millis(reader.next_nanos(), reader.next_calls()),
            nanos_to_millis(reader.p95_next_nanos()),
            nanos_to_millis(reader.max_next_nanos()),
            reader.next_calls(),
            reader.eof_calls(),
            average_rows_per_batch(reader.output_rows(), reader.output_batches()),
            nanos_to_millis(consume_nanos),
            reader.compressed_bytes_scanned(),
            reader.compressed_bytes_total(),
        );
    }
    let Some(output) = finished? else {
        return Ok(None);
    };
    Ok(Some(ParquetMapChunkResult { output, metrics }))
}

#[allow(clippy::too_many_arguments)]
fn parquet_row_group_map_chunk_view<State, Output, ConsumeBatch, Finish>(
    path: PathBuf,
    batch_size: usize,
    row_groups: Vec<usize>,
    projection: &Projection,
    metadata_cache: &ParquetMetadataCache,
    file_cache: Arc<ParquetFileCache>,
    object_store: &dyn ObjectStore,
    mut state: State,
    mut consume_batch: ConsumeBatch,
    finish: Finish,
    profile_label: Option<&str>,
    profile_chunk: usize,
) -> Result<Option<ParquetMapChunkResult<Output>>>
where
    ConsumeBatch: for<'a> FnMut(BatchView<'a>, &mut State) -> Result<Option<()>>,
    Finish: FnOnce(State) -> Result<Option<Output>>,
{
    let started = profile_label.map(|_| Instant::now());
    let mut reader = ParquetBatchReader::try_new_with_row_groups(
        path,
        batch_size,
        projection,
        row_groups.clone(),
        metadata_cache,
        file_cache,
        object_store,
    )?;
    let reader_setup_nanos = started
        .map(|started| elapsed_nanos(started.elapsed()))
        .unwrap_or_default();
    let mut batches = 0_usize;
    let mut rows = 0_usize;
    let mut read_nanos = 0_u64;
    let mut consume_nanos = 0_u64;
    loop {
        let read_start = Instant::now();
        let Some(batch) = reader.next() else {
            read_nanos = read_nanos.saturating_add(elapsed_nanos(read_start.elapsed()));
            break;
        };
        let batch = batch?;
        read_nanos = read_nanos.saturating_add(elapsed_nanos(read_start.elapsed()));
        batches += 1;
        rows += batch.num_rows();
        let consume_start = Instant::now();
        if consume_batch(BatchView::new(&batch), &mut state)?.is_none() {
            return Ok(None);
        }
        consume_nanos = consume_nanos.saturating_add(elapsed_nanos(consume_start.elapsed()));
    }
    let total_nanos = started
        .map(|started| elapsed_nanos(started.elapsed()))
        .unwrap_or_default();
    let metrics = ParquetMapChunkMetrics {
        chunks: 1,
        row_groups: row_groups.len(),
        projected_columns: reader.projected_columns(),
        rows,
        batches,
        zero_batches: reader.zero_row_batches(),
        total_nanos,
        read_nanos,
        reader_next_nanos: reader.next_nanos(),
        reader_p95_next_nanos: reader.p95_next_nanos(),
        reader_max_next_nanos: reader.max_next_nanos(),
        reader_calls: reader.next_calls(),
        reader_eof: reader.eof_calls(),
        consume_nanos,
        compressed_bytes_scanned: reader.compressed_bytes_scanned(),
        compressed_bytes_total: reader.compressed_bytes_total(),
    };
    let finished = finish(state);
    if let (Some(label), Some(_)) = (profile_label, started) {
        eprintln!(
            "[dodam:tpch-profile] row_group_map_view {label}: chunk={} row_groups={} projected_columns={} rows={} batches={} zero_batches={} total={:.3} ms setup={:.3} ms metadata={:.3} ms planning={:.3} ms read_next={:.3} ms reader_next={:.3} ms reader_next_avg={:.3} ms reader_next_p95={:.3} ms reader_next_max={:.3} ms reader_calls={} reader_eof={} avg_batch_rows={:.1} consume={:.3} ms compressed={}/{}",
            profile_chunk,
            row_groups.len(),
            reader.projected_columns(),
            rows,
            batches,
            reader.zero_row_batches(),
            nanos_to_millis(metrics.total_nanos),
            nanos_to_millis(reader_setup_nanos),
            nanos_to_millis(reader.metadata_nanos()),
            nanos_to_millis(reader.planning_nanos()),
            nanos_to_millis(read_nanos),
            nanos_to_millis(reader.next_nanos()),
            average_nanos_millis(reader.next_nanos(), reader.next_calls()),
            nanos_to_millis(reader.p95_next_nanos()),
            nanos_to_millis(reader.max_next_nanos()),
            reader.next_calls(),
            reader.eof_calls(),
            average_rows_per_batch(reader.output_rows(), reader.output_batches()),
            nanos_to_millis(consume_nanos),
            reader.compressed_bytes_scanned(),
            reader.compressed_bytes_total(),
        );
    }
    let Some(output) = finished? else {
        return Ok(None);
    };
    Ok(Some(ParquetMapChunkResult { output, metrics }))
}

#[allow(clippy::too_many_arguments)]
fn parquet_row_group_map_dictionary_chunk<State, Output, ConsumeBatch, Finish>(
    path: PathBuf,
    batch_size: usize,
    row_groups: Vec<usize>,
    projection: &Projection,
    dictionary_columns: &[String],
    metadata_cache: &ParquetMetadataCache,
    file_cache: Arc<ParquetFileCache>,
    object_store: &dyn ObjectStore,
    mut state: State,
    mut consume_batch: ConsumeBatch,
    finish: Finish,
    profile_label: Option<&str>,
    profile_chunk: usize,
) -> Result<Option<ParquetMapChunkResult<Output>>>
where
    ConsumeBatch: FnMut(RecordBatch, &mut State) -> Result<Option<()>>,
    Finish: FnOnce(State) -> Result<Option<Output>>,
{
    let started = profile_label.map(|_| Instant::now());
    let mut reader = ParquetBatchReader::try_new_with_row_groups_dictionary_columns(
        path,
        batch_size,
        projection,
        row_groups.clone(),
        dictionary_columns,
        metadata_cache,
        file_cache,
        object_store,
    )?;
    let reader_setup_nanos = started
        .map(|started| elapsed_nanos(started.elapsed()))
        .unwrap_or_default();
    let mut batches = 0_usize;
    let mut rows = 0_usize;
    let mut read_nanos = 0_u64;
    let mut consume_nanos = 0_u64;
    loop {
        let read_start = Instant::now();
        let Some(batch) = reader.next() else {
            read_nanos = read_nanos.saturating_add(elapsed_nanos(read_start.elapsed()));
            break;
        };
        let batch = batch?;
        read_nanos = read_nanos.saturating_add(elapsed_nanos(read_start.elapsed()));
        batches += 1;
        rows += batch.num_rows();
        let consume_start = Instant::now();
        if consume_batch(batch, &mut state)?.is_none() {
            return Ok(None);
        }
        consume_nanos = consume_nanos.saturating_add(elapsed_nanos(consume_start.elapsed()));
    }
    let total_nanos = started
        .map(|started| elapsed_nanos(started.elapsed()))
        .unwrap_or_default();
    let metrics = ParquetMapChunkMetrics {
        chunks: 1,
        row_groups: row_groups.len(),
        projected_columns: reader.projected_columns(),
        rows,
        batches,
        zero_batches: reader.zero_row_batches(),
        total_nanos,
        read_nanos,
        reader_next_nanos: reader.next_nanos(),
        reader_p95_next_nanos: reader.p95_next_nanos(),
        reader_max_next_nanos: reader.max_next_nanos(),
        reader_calls: reader.next_calls(),
        reader_eof: reader.eof_calls(),
        consume_nanos,
        compressed_bytes_scanned: reader.compressed_bytes_scanned(),
        compressed_bytes_total: reader.compressed_bytes_total(),
    };
    let finished = finish(state);
    if let (Some(label), Some(_)) = (profile_label, started) {
        eprintln!(
            "[dodam:tpch-profile] dictionary_row_group_map {label}: chunk={} row_groups={} projected_columns={} rows={} batches={} zero_batches={} total={:.3} ms setup={:.3} ms metadata={:.3} ms planning={:.3} ms read_next={:.3} ms reader_next={:.3} ms reader_next_avg={:.3} ms reader_next_p95={:.3} ms reader_next_max={:.3} ms reader_calls={} reader_eof={} avg_batch_rows={:.1} consume={:.3} ms compressed={}/{}",
            profile_chunk,
            row_groups.len(),
            reader.projected_columns(),
            rows,
            batches,
            reader.zero_row_batches(),
            nanos_to_millis(metrics.total_nanos),
            nanos_to_millis(reader_setup_nanos),
            nanos_to_millis(reader.metadata_nanos()),
            nanos_to_millis(reader.planning_nanos()),
            nanos_to_millis(read_nanos),
            nanos_to_millis(reader.next_nanos()),
            average_nanos_millis(reader.next_nanos(), reader.next_calls()),
            nanos_to_millis(reader.p95_next_nanos()),
            nanos_to_millis(reader.max_next_nanos()),
            reader.next_calls(),
            reader.eof_calls(),
            average_rows_per_batch(reader.output_rows(), reader.output_batches()),
            nanos_to_millis(consume_nanos),
            reader.compressed_bytes_scanned(),
            reader.compressed_bytes_total(),
        );
    }
    let Some(output) = finished? else {
        return Ok(None);
    };
    Ok(Some(ParquetMapChunkResult { output, metrics }))
}

#[allow(clippy::too_many_arguments)]
fn parquet_row_group_map_dictionary_chunk_view<State, Output, ConsumeBatch, Finish>(
    path: PathBuf,
    batch_size: usize,
    row_groups: Vec<usize>,
    projection: &Projection,
    dictionary_columns: &[String],
    metadata_cache: &ParquetMetadataCache,
    file_cache: Arc<ParquetFileCache>,
    object_store: &dyn ObjectStore,
    mut state: State,
    mut consume_batch: ConsumeBatch,
    finish: Finish,
    profile_label: Option<&str>,
    profile_chunk: usize,
) -> Result<Option<ParquetMapChunkResult<Output>>>
where
    ConsumeBatch: for<'a> FnMut(BatchView<'a>, &mut State) -> Result<Option<()>>,
    Finish: FnOnce(State) -> Result<Option<Output>>,
{
    let started = profile_label.map(|_| Instant::now());
    let mut reader = ParquetBatchReader::try_new_with_row_groups_dictionary_columns(
        path,
        batch_size,
        projection,
        row_groups.clone(),
        dictionary_columns,
        metadata_cache,
        file_cache,
        object_store,
    )?;
    let reader_setup_nanos = started
        .map(|started| elapsed_nanos(started.elapsed()))
        .unwrap_or_default();
    let mut batches = 0_usize;
    let mut rows = 0_usize;
    let mut read_nanos = 0_u64;
    let mut consume_nanos = 0_u64;
    loop {
        let read_start = Instant::now();
        let Some(batch) = reader.next() else {
            read_nanos = read_nanos.saturating_add(elapsed_nanos(read_start.elapsed()));
            break;
        };
        let batch = batch?;
        read_nanos = read_nanos.saturating_add(elapsed_nanos(read_start.elapsed()));
        batches += 1;
        rows += batch.num_rows();
        let consume_start = Instant::now();
        if consume_batch(BatchView::new(&batch), &mut state)?.is_none() {
            return Ok(None);
        }
        consume_nanos = consume_nanos.saturating_add(elapsed_nanos(consume_start.elapsed()));
    }
    let total_nanos = started
        .map(|started| elapsed_nanos(started.elapsed()))
        .unwrap_or_default();
    let metrics = ParquetMapChunkMetrics {
        chunks: 1,
        row_groups: row_groups.len(),
        projected_columns: reader.projected_columns(),
        rows,
        batches,
        zero_batches: reader.zero_row_batches(),
        total_nanos,
        read_nanos,
        reader_next_nanos: reader.next_nanos(),
        reader_p95_next_nanos: reader.p95_next_nanos(),
        reader_max_next_nanos: reader.max_next_nanos(),
        reader_calls: reader.next_calls(),
        reader_eof: reader.eof_calls(),
        consume_nanos,
        compressed_bytes_scanned: reader.compressed_bytes_scanned(),
        compressed_bytes_total: reader.compressed_bytes_total(),
    };
    let finished = finish(state);
    if let (Some(label), Some(_)) = (profile_label, started) {
        eprintln!(
            "[dodam:tpch-profile] dictionary_row_group_map_view {label}: chunk={} row_groups={} projected_columns={} rows={} batches={} zero_batches={} total={:.3} ms setup={:.3} ms metadata={:.3} ms planning={:.3} ms read_next={:.3} ms reader_next={:.3} ms reader_next_avg={:.3} ms reader_next_p95={:.3} ms reader_next_max={:.3} ms reader_calls={} reader_eof={} avg_batch_rows={:.1} consume={:.3} ms compressed={}/{}",
            profile_chunk,
            row_groups.len(),
            reader.projected_columns(),
            rows,
            batches,
            reader.zero_row_batches(),
            nanos_to_millis(metrics.total_nanos),
            nanos_to_millis(reader_setup_nanos),
            nanos_to_millis(reader.metadata_nanos()),
            nanos_to_millis(reader.planning_nanos()),
            nanos_to_millis(read_nanos),
            nanos_to_millis(reader.next_nanos()),
            average_nanos_millis(reader.next_nanos(), reader.next_calls()),
            nanos_to_millis(reader.p95_next_nanos()),
            nanos_to_millis(reader.max_next_nanos()),
            reader.next_calls(),
            reader.eof_calls(),
            average_rows_per_batch(reader.output_rows(), reader.output_batches()),
            nanos_to_millis(consume_nanos),
            reader.compressed_bytes_scanned(),
            reader.compressed_bytes_total(),
        );
    }
    let Some(output) = finished? else {
        return Ok(None);
    };
    Ok(Some(ParquetMapChunkResult { output, metrics }))
}

fn log_late_materialized_metrics(label: &str, metrics: LateMaterializedMetrics, chunks: usize) {
    if std::env::var_os("DODAM_TPCH_PROFILE").is_none() {
        return;
    }
    let ratio = metrics.selected_ratio();
    let selector_runs_per_selected = if metrics.selected_rows == 0 {
        0.0
    } else {
        metrics.selector_runs as f64 / metrics.selected_rows as f64
    };
    eprintln!(
        "[dodam:tpch-profile] {label}: late_materialized rows={} selected={} ratio={:.6} selector_runs={} chunks={chunks} predicate_read={:.3} ms payload_read={:.3} ms predicate_batches={} payload_batches={} payload_rows={} selector_runs_per_selected={:.6}",
        metrics.total_rows,
        metrics.selected_rows,
        ratio,
        metrics.selector_runs,
        metrics.predicate_read_nanos as f64 / 1_000_000.0,
        metrics.payload_read_nanos as f64 / 1_000_000.0,
        metrics.predicate_batches,
        metrics.payload_batches,
        metrics.payload_rows,
        selector_runs_per_selected
    );
    let status = if metrics.selected_rows > 0
        && metrics.selector_runs as f64 / metrics.selected_rows as f64 > 0.50
    {
        "fragmented_late_materialized"
    } else {
        "late_materialized_blocked_path"
    };
    eprintln!(
        "[dodam:physical] kind=late_materialized status={status} rows={} selected={} selected_ratio={:.6} selector_runs={} selector_runs_per_selected={:.6} chunks={chunks} predicate_read_ms={:.3} payload_read_ms={:.3} predicate_batches={} payload_batches={} payload_rows={} label={label}",
        metrics.total_rows,
        metrics.selected_rows,
        ratio,
        metrics.selector_runs,
        selector_runs_per_selected,
        metrics.predicate_read_nanos as f64 / 1_000_000.0,
        metrics.payload_read_nanos as f64 / 1_000_000.0,
        metrics.predicate_batches,
        metrics.payload_batches,
        metrics.payload_rows
    );
}

fn log_parquet_map_summary(kind: &str, label: Option<&str>, metrics: ParquetMapChunkMetrics) {
    let Some(label) = label else {
        return;
    };
    eprintln!(
        "[dodam:scan-profile] {kind}_summary {label}: chunks={} row_groups={} projected_columns={} rows={} batches={} zero_batches={} total_sum={:.3} ms read_next={:.3} ms reader_next={:.3} ms reader_next_avg={:.3} ms reader_next_p95_max={:.3} ms reader_next_max={:.3} ms reader_calls={} reader_eof={} avg_batch_rows={:.1} consume={:.3} ms compressed={}/{}",
        metrics.chunks,
        metrics.row_groups,
        metrics.projected_columns,
        metrics.rows,
        metrics.batches,
        metrics.zero_batches,
        nanos_to_millis(metrics.total_nanos),
        nanos_to_millis(metrics.read_nanos),
        nanos_to_millis(metrics.reader_next_nanos),
        average_nanos_millis(metrics.reader_next_nanos, metrics.reader_calls),
        nanos_to_millis(metrics.reader_p95_next_nanos),
        nanos_to_millis(metrics.reader_max_next_nanos),
        metrics.reader_calls,
        metrics.reader_eof,
        average_rows_per_batch(metrics.rows, metrics.batches),
        nanos_to_millis(metrics.consume_nanos),
        metrics.compressed_bytes_scanned,
        metrics.compressed_bytes_total,
    );
    let total = metrics.total_nanos as f64;
    let read_fraction = if total > 0.0 {
        metrics.read_nanos as f64 / total
    } else {
        0.0
    };
    let consume_fraction = if total > 0.0 {
        metrics.consume_nanos as f64 / total
    } else {
        0.0
    };
    let bottleneck = if read_fraction >= 0.60 {
        if metrics.projected_columns <= 4 {
            "read-heavy-narrow"
        } else if metrics.projected_columns >= 8 {
            "read-heavy-wide"
        } else {
            "read-heavy"
        }
    } else if consume_fraction >= 0.50 {
        "consume-heavy"
    } else {
        "mixed"
    };
    eprintln!(
        "[dodam:physical] kind={kind}_summary status=blocked_row_group_map bottleneck={bottleneck} read_fraction={read_fraction:.6} consume_fraction={consume_fraction:.6} rows={} batches={} compressed_ratio={:.6} label={label}",
        metrics.rows,
        metrics.batches,
        if metrics.compressed_bytes_total > 0 {
            metrics.compressed_bytes_scanned as f64 / metrics.compressed_bytes_total as f64
        } else {
            0.0
        },
    );
}

struct BoolLookupDiscountedRevenueState {
    lookup: Arc<DenseI64BoolLookup>,
    matched: f64,
    total: f64,
}

fn build_date32_range_selection_batch(
    batch: RecordBatch,
    start_days: i32,
    end_days: i32,
    selection: &mut LateSelectionBuilder,
) -> Result<Option<()>> {
    let shipdate_index = physical_batch_column_index(&batch, "l_shipdate")?;
    let Some(shipdates) = batch
        .column(shipdate_index)
        .as_any()
        .downcast_ref::<Date32Array>()
    else {
        return Ok(None);
    };
    if shipdates.null_count() != 0 {
        return Ok(None);
    }
    for &shipdate in shipdates.values().as_ref() {
        selection.push(shipdate >= start_days && shipdate < end_days);
    }
    Ok(Some(()))
}

fn build_date32_range_selection_view(
    view: BatchView<'_>,
    start_days: i32,
    end_days: i32,
    selection: &mut LateSelectionBuilder,
) -> Result<Option<()>> {
    if view.num_columns() == 1 {
        let Some(shipdates) = view.date32_vector(0) else {
            return Ok(None);
        };
        let Some(shipdate_values) = shipdates.values_if_null_free() else {
            return Ok(None);
        };
        for &shipdate in shipdate_values {
            selection.push(shipdate >= start_days && shipdate < end_days);
        }
        return Ok(Some(()));
    }
    let Some(batch) = view.try_record_batch() else {
        return Ok(None);
    };
    build_date32_range_selection_batch(batch.clone(), start_days, end_days, selection)
}

fn consume_discounted_revenue_by_i64_bool_lookup_batch(
    batch: RecordBatch,
    state: &mut BoolLookupDiscountedRevenueState,
) -> Result<Option<()>> {
    let partkey_index = physical_batch_column_index(&batch, "l_partkey")?;
    let extendedprice_index = physical_batch_column_index(&batch, "l_extendedprice")?;
    let discount_index = physical_batch_column_index(&batch, "l_discount")?;
    let Some(partkeys) = batch
        .column(partkey_index)
        .as_any()
        .downcast_ref::<Int64Array>()
    else {
        return Ok(None);
    };
    let Some(extendedprices) = batch
        .column(extendedprice_index)
        .as_any()
        .downcast_ref::<Decimal128Array>()
    else {
        return Ok(None);
    };
    let Some(discounts) = batch
        .column(discount_index)
        .as_any()
        .downcast_ref::<Decimal128Array>()
    else {
        return Ok(None);
    };
    if partkeys.null_count() != 0 || extendedprices.null_count() != 0 || discounts.null_count() != 0
    {
        return Ok(None);
    }
    consume_discounted_revenue_by_i64_bool_lookup_arrays(partkeys, extendedprices, discounts, state)
}

fn consume_discounted_revenue_by_i64_bool_lookup_view(
    view: BatchView<'_>,
    state: &mut BoolLookupDiscountedRevenueState,
) -> Result<Option<()>> {
    if view.num_columns() == 3 {
        let (Some(partkeys), Some(extendedprices), Some(discounts)) = (
            view.i64_vector(0),
            view.decimal128_vector(1),
            view.decimal128_vector(2),
        ) else {
            return Ok(None);
        };
        return consume_discounted_revenue_by_i64_bool_lookup_vectors(
            partkeys,
            extendedprices,
            discounts,
            state,
        );
    }
    let Some(batch) = view.try_record_batch() else {
        return Ok(None);
    };
    consume_discounted_revenue_by_i64_bool_lookup_batch(batch.clone(), state)
}

fn consume_discounted_revenue_by_i64_bool_lookup_vectors(
    partkeys: I64VectorView<'_>,
    extendedprices: Decimal128VectorView<'_>,
    discounts: Decimal128VectorView<'_>,
    state: &mut BoolLookupDiscountedRevenueState,
) -> Result<Option<()>> {
    let (Some(partkey_values), Some(price_scale), Some(discount_scale)) = (
        partkeys.values_if_null_free(),
        extendedprices.scale_i64(),
        discounts.scale_i64(),
    ) else {
        return Ok(None);
    };
    if extendedprices.null_count() != 0
        || discounts.null_count() != 0
        || extendedprices.precision() > 18
        || discounts.precision() > 18
    {
        return Ok(None);
    }
    let revenue_scale = 1.0 / ((price_scale as f64) * (discount_scale as f64));
    let extendedprice_values = extendedprices.raw_values();
    let discount_values = discounts.raw_values();
    for row in 0..partkey_values.len() {
        let Some(is_promo) = state.lookup.get(partkey_values[row]) else {
            continue;
        };
        let value = ((extendedprice_values[row] as i64)
            * (discount_scale - discount_values[row] as i64)) as f64
            * revenue_scale;
        if is_promo {
            state.matched += value;
        }
        state.total += value;
    }
    Ok(Some(()))
}

fn consume_discounted_revenue_by_i64_bool_lookup_arrays(
    partkeys: &Int64Array,
    extendedprices: &Decimal128Array,
    discounts: &Decimal128Array,
    state: &mut BoolLookupDiscountedRevenueState,
) -> Result<Option<()>> {
    if partkeys.null_count() != 0 || extendedprices.null_count() != 0 || discounts.null_count() != 0
    {
        return Ok(None);
    }
    let (price_precision, price_decimal_scale) = decimal128_precision_scale(extendedprices)?;
    let (discount_precision, discount_decimal_scale) = decimal128_precision_scale(discounts)?;
    if price_precision > 18 || discount_precision > 18 {
        return Ok(None);
    }
    let price_scale = decimal_scale_i64(price_decimal_scale)?;
    let discount_scale = decimal_scale_i64(discount_decimal_scale)?;
    let revenue_scale = 1.0 / ((price_scale as f64) * (discount_scale as f64));
    for row in 0..partkeys.len() {
        let Some(is_promo) = state.lookup.get(partkeys.value(row)) else {
            continue;
        };
        let value = ((extendedprices.values()[row] as i64)
            * (discount_scale - discounts.values()[row] as i64)) as f64
            * revenue_scale;
        if is_promo {
            state.matched += value;
        }
        state.total += value;
    }
    Ok(Some(()))
}

struct SelectedDiscountRevenueState {
    selected_discounts: Vec<i64>,
    discount_scale: Option<i64>,
    extendedprice_scale: Option<i64>,
    discount_offset: usize,
    revenue: f64,
}

#[allow(clippy::too_many_arguments)]
fn build_date32_discount_quantity_selection_batch(
    batch: RecordBatch,
    start_days: i32,
    end_days: i32,
    discount_low: f64,
    discount_high: f64,
    quantity_limit: f64,
    selection: &mut LateSelectionBuilder,
    state: &mut SelectedDiscountRevenueState,
) -> Result<Option<()>> {
    let shipdate_index = physical_batch_column_index(&batch, "l_shipdate")?;
    let discount_index = physical_batch_column_index(&batch, "l_discount")?;
    let quantity_index = physical_batch_column_index(&batch, "l_quantity")?;
    let Some(shipdates) = batch
        .column(shipdate_index)
        .as_any()
        .downcast_ref::<Date32Array>()
    else {
        return Ok(None);
    };
    let Some(discounts) = batch
        .column(discount_index)
        .as_any()
        .downcast_ref::<Decimal128Array>()
    else {
        return Ok(None);
    };
    let Some(quantities) = batch
        .column(quantity_index)
        .as_any()
        .downcast_ref::<Decimal128Array>()
    else {
        return Ok(None);
    };
    if shipdates.null_count() != 0 || discounts.null_count() != 0 || quantities.null_count() != 0 {
        return Ok(None);
    }
    let (discount_precision, discount_decimal_scale) = decimal128_precision_scale(discounts)?;
    let (quantity_precision, quantity_decimal_scale) = decimal128_precision_scale(quantities)?;
    if discount_precision > 18 || quantity_precision > 18 {
        return Ok(None);
    }
    let discount_scale_value = decimal_scale_i64(discount_decimal_scale)?;
    let quantity_scale_value = decimal_scale_i64(quantity_decimal_scale)?;
    if let Some(existing) = state.discount_scale {
        if existing != discount_scale_value {
            return Ok(None);
        }
    } else {
        state.discount_scale = Some(discount_scale_value);
    }
    let discount_low_raw = scaled_f64_to_i64(discount_low, discount_scale_value);
    let discount_high_raw = scaled_f64_to_i64(discount_high, discount_scale_value);
    let quantity_limit_raw = scaled_f64_to_i64(quantity_limit, quantity_scale_value);
    let shipdate_values = shipdates.values().as_ref();
    let discount_values = discounts.values();
    let quantity_values = quantities.values();
    for row in 0..shipdate_values.len() {
        let shipdate = shipdate_values[row];
        let discount = discount_values[row] as i64;
        let selected = shipdate >= start_days
            && shipdate < end_days
            && discount >= discount_low_raw
            && discount <= discount_high_raw
            && (quantity_values[row] as i64) < quantity_limit_raw;
        if selected {
            state.selected_discounts.push(discount);
        }
        selection.push(selected);
    }
    Ok(Some(()))
}

#[allow(clippy::too_many_arguments)]
fn build_date32_discount_quantity_selection_view(
    view: BatchView<'_>,
    start_days: i32,
    end_days: i32,
    discount_low: f64,
    discount_high: f64,
    quantity_limit: f64,
    selection: &mut LateSelectionBuilder,
    state: &mut SelectedDiscountRevenueState,
) -> Result<Option<()>> {
    if view.num_columns() == 3 {
        let (Some(shipdates), Some(discounts), Some(quantities)) = (
            view.date32_vector(0),
            view.decimal128_vector(1),
            view.decimal128_vector(2),
        ) else {
            return Ok(None);
        };
        return build_date32_discount_quantity_selection_vectors(
            shipdates,
            discounts,
            quantities,
            start_days,
            end_days,
            discount_low,
            discount_high,
            quantity_limit,
            selection,
            state,
        );
    }
    let Some(batch) = view.try_record_batch() else {
        return Ok(None);
    };
    build_date32_discount_quantity_selection_batch(
        batch.clone(),
        start_days,
        end_days,
        discount_low,
        discount_high,
        quantity_limit,
        selection,
        state,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_date32_discount_quantity_selection_vectors(
    shipdates: Date32VectorView<'_>,
    discounts: Decimal128VectorView<'_>,
    quantities: Decimal128VectorView<'_>,
    start_days: i32,
    end_days: i32,
    discount_low: f64,
    discount_high: f64,
    quantity_limit: f64,
    selection: &mut LateSelectionBuilder,
    state: &mut SelectedDiscountRevenueState,
) -> Result<Option<()>> {
    let (Some(shipdate_values), Some(discount_scale_value), Some(quantity_scale_value)) = (
        shipdates.values_if_null_free(),
        discounts.scale_i64(),
        quantities.scale_i64(),
    ) else {
        return Ok(None);
    };
    if discounts.null_count() != 0
        || quantities.null_count() != 0
        || discounts.precision() > 18
        || quantities.precision() > 18
    {
        return Ok(None);
    }
    if let Some(existing) = state.discount_scale {
        if existing != discount_scale_value {
            return Ok(None);
        }
    } else {
        state.discount_scale = Some(discount_scale_value);
    }
    let discount_low_raw = scaled_f64_to_i64(discount_low, discount_scale_value);
    let discount_high_raw = scaled_f64_to_i64(discount_high, discount_scale_value);
    let quantity_limit_raw = scaled_f64_to_i64(quantity_limit, quantity_scale_value);
    let discount_values = discounts.raw_values();
    let quantity_values = quantities.raw_values();
    for row in 0..shipdate_values.len() {
        let shipdate = shipdate_values[row];
        let discount = discount_values[row] as i64;
        let selected = shipdate >= start_days
            && shipdate < end_days
            && discount >= discount_low_raw
            && discount <= discount_high_raw
            && (quantity_values[row] as i64) < quantity_limit_raw;
        if selected {
            state.selected_discounts.push(discount);
        }
        selection.push(selected);
    }
    Ok(Some(()))
}

fn consume_selected_discount_revenue_batch(
    batch: RecordBatch,
    state: &mut SelectedDiscountRevenueState,
) -> Result<Option<()>> {
    let extendedprice_index = physical_batch_column_index(&batch, "l_extendedprice")?;
    let Some(extendedprices) = batch
        .column(extendedprice_index)
        .as_any()
        .downcast_ref::<Decimal128Array>()
    else {
        return Ok(None);
    };
    if extendedprices.null_count() != 0 {
        return Ok(None);
    }
    let (precision, decimal_scale) = decimal128_precision_scale(extendedprices)?;
    if precision > 18 {
        return Ok(None);
    }
    let price_scale = decimal_scale_i64(decimal_scale)?;
    if let Some(existing) = state.extendedprice_scale {
        if existing != price_scale {
            return Ok(None);
        }
    } else {
        state.extendedprice_scale = Some(price_scale);
    }
    let discount_scale = state.discount_scale.ok_or_else(|| {
        DodamError::UnsupportedSql("selected discount revenue missing discount scale".to_string())
    })?;
    let revenue_scale = 1.0 / ((price_scale as f64) * (discount_scale as f64));
    for &extendedprice in extendedprices.values() {
        let discount = *state
            .selected_discounts
            .get(state.discount_offset)
            .ok_or_else(|| {
                DodamError::UnsupportedSql("selected discount revenue payload mismatch".to_string())
            })?;
        state.revenue += ((extendedprice as i64) * discount) as f64 * revenue_scale;
        state.discount_offset += 1;
    }
    Ok(Some(()))
}

fn consume_selected_discount_revenue_view(
    view: BatchView<'_>,
    state: &mut SelectedDiscountRevenueState,
) -> Result<Option<()>> {
    if view.num_columns() == 1 {
        let Some(extendedprices) = view.decimal128_vector(0) else {
            return Ok(None);
        };
        return consume_selected_discount_revenue_vectors(extendedprices, state);
    }
    let Some(batch) = view.try_record_batch() else {
        return Ok(None);
    };
    consume_selected_discount_revenue_batch(batch.clone(), state)
}

fn consume_selected_discount_revenue_vectors(
    extendedprices: Decimal128VectorView<'_>,
    state: &mut SelectedDiscountRevenueState,
) -> Result<Option<()>> {
    if extendedprices.null_count() != 0 || extendedprices.precision() > 18 {
        return Ok(None);
    }
    let Some(price_scale) = extendedprices.scale_i64() else {
        return Ok(None);
    };
    if let Some(existing) = state.extendedprice_scale {
        if existing != price_scale {
            return Ok(None);
        }
    } else {
        state.extendedprice_scale = Some(price_scale);
    }
    let discount_scale = state.discount_scale.ok_or_else(|| {
        DodamError::UnsupportedSql("selected discount revenue missing discount scale".to_string())
    })?;
    let revenue_scale = 1.0 / ((price_scale as f64) * (discount_scale as f64));
    for &extendedprice in extendedprices.raw_values() {
        let discount = *state
            .selected_discounts
            .get(state.discount_offset)
            .ok_or_else(|| {
                DodamError::UnsupportedSql("selected discount revenue payload mismatch".to_string())
            })?;
        state.revenue += ((extendedprice as i64) * discount) as f64 * revenue_scale;
        state.discount_offset += 1;
    }
    Ok(Some(()))
}

fn push_late_selector_run(
    selectors: &mut Vec<RowSelector>,
    current_selected: &mut Option<bool>,
    run_len: &mut usize,
    selected: bool,
) {
    append_late_selector_run(selectors, current_selected, run_len, selected, 1);
}

fn append_late_selector_run(
    selectors: &mut Vec<RowSelector>,
    current_selected: &mut Option<bool>,
    run_len: &mut usize,
    selected: bool,
    len: usize,
) {
    if len == 0 {
        return;
    }
    match *current_selected {
        Some(current) if current == selected => *run_len += len,
        Some(current) => {
            selectors.push(if current {
                RowSelector::select(*run_len)
            } else {
                RowSelector::skip(*run_len)
            });
            *current_selected = Some(selected);
            *run_len = len;
        }
        None => {
            *current_selected = Some(selected);
            *run_len = len;
        }
    }
}

fn finish_late_selector_run(
    selectors: &mut Vec<RowSelector>,
    current_selected: Option<bool>,
    run_len: usize,
) {
    if run_len == 0 {
        return;
    }
    if current_selected.unwrap_or(false) {
        selectors.push(RowSelector::select(run_len));
    } else {
        selectors.push(RowSelector::skip(run_len));
    }
}

fn decimal128_precision_scale(values: &Decimal128Array) -> Result<(u8, i8)> {
    match values.data_type() {
        DataType::Decimal128(precision, scale) => Ok((*precision, *scale)),
        _ => Ok((0, 0)),
    }
}

fn decimal_scale_i64(scale: i8) -> Result<i64> {
    let scale = u32::try_from(scale).map_err(|_| {
        DodamError::UnsupportedSql(format!("negative decimal scale {scale} is unsupported"))
    })?;
    10_i64
        .checked_pow(scale)
        .ok_or_else(|| DodamError::UnsupportedSql(format!("decimal scale {scale} overflows i64")))
}

fn scaled_f64_to_i64(value: f64, scale: i64) -> i64 {
    (value * scale as f64).round() as i64
}

#[allow(clippy::too_many_arguments)]
fn scan_i64_set_filtered_row_groups(
    path: PathBuf,
    batch_size: usize,
    projection: &Projection,
    row_groups: Vec<usize>,
    filter_column: &str,
    keys: Arc<HashSet<i64>>,
    metadata_cache: &ParquetMetadataCache,
    file_cache: Arc<ParquetFileCache>,
    object_store: &dyn ObjectStore,
) -> Result<Vec<RecordBatch>> {
    let profile_label =
        scan_profile_enabled().then(|| parquet_map_profile_label(&path, projection));
    let row_group_count = row_groups.len();
    let started = Instant::now();
    let setup_started = Instant::now();
    let mut reader = ParquetBatchReader::try_new_with_row_groups_i64_set_filter(
        path,
        batch_size,
        projection,
        row_groups,
        filter_column,
        keys,
        metadata_cache,
        file_cache,
        object_store,
    )?;
    let setup_nanos = elapsed_nanos(setup_started.elapsed());
    let mut batches = Vec::new();
    let mut read_loop_nanos = 0_u64;
    let mut output_rows = 0_usize;
    loop {
        let next_started = Instant::now();
        let next = reader.next();
        read_loop_nanos = read_loop_nanos.saturating_add(elapsed_nanos(next_started.elapsed()));
        let Some(batch) = next else {
            break;
        };
        let batch = batch?;
        if batch.num_rows() > 0 {
            output_rows = output_rows.saturating_add(batch.num_rows());
            batches.push(batch);
        }
    }
    if let Some(label) = profile_label {
        eprintln!(
            "[dodam:scan-profile] i64_set_filter {label}: elapsed={:.3} ms setup={:.3} ms read_loop={:.3} ms row_groups={}/{} projected_columns={} rows={} batches={} zero_batches={} bytes={} metadata={:.3} ms planning={:.3} ms parquet_next={:.3} ms parquet_next_avg={:.3} ms parquet_next_max={:.3} ms parquet_calls={} parquet_eof={} parquet_rows={} parquet_batches={} avg_batch_rows={:.1}",
            started.elapsed().as_secs_f64() * 1000.0,
            nanos_to_millis(setup_nanos),
            nanos_to_millis(read_loop_nanos),
            row_group_count,
            reader.row_groups_total(),
            reader.projected_columns(),
            output_rows,
            batches.len(),
            reader.zero_row_batches(),
            reader.compressed_bytes_scanned(),
            nanos_to_millis(reader.metadata_nanos()),
            nanos_to_millis(reader.planning_nanos()),
            nanos_to_millis(reader.next_nanos()),
            average_nanos_millis(reader.next_nanos(), reader.next_calls()),
            nanos_to_millis(reader.max_next_nanos()),
            reader.next_calls(),
            reader.eof_calls(),
            reader.output_rows(),
            reader.output_batches(),
            average_rows_per_batch(reader.output_rows(), reader.output_batches()),
        );
    }
    Ok(batches)
}

#[allow(clippy::too_many_arguments)]
fn scan_i64_bloom_filtered_row_groups(
    path: PathBuf,
    batch_size: usize,
    projection: &Projection,
    row_groups: Vec<usize>,
    filter_column: &str,
    bloom: Arc<I64BloomPredicate>,
    metadata_cache: &ParquetMetadataCache,
    file_cache: Arc<ParquetFileCache>,
    object_store: &dyn ObjectStore,
) -> Result<Vec<RecordBatch>> {
    let profile_label =
        scan_profile_enabled().then(|| parquet_map_profile_label(&path, projection));
    let row_group_count = row_groups.len();
    let started = Instant::now();
    let setup_started = Instant::now();
    let mut reader = ParquetBatchReader::try_new_with_row_groups_i64_bloom_filter(
        path,
        batch_size,
        projection,
        row_groups,
        filter_column,
        bloom,
        metadata_cache,
        file_cache,
        object_store,
    )?;
    let setup_nanos = elapsed_nanos(setup_started.elapsed());
    let mut batches = Vec::new();
    let mut read_loop_nanos = 0_u64;
    let mut output_rows = 0_usize;
    loop {
        let next_started = Instant::now();
        let next = reader.next();
        read_loop_nanos = read_loop_nanos.saturating_add(elapsed_nanos(next_started.elapsed()));
        let Some(batch) = next else {
            break;
        };
        let batch = batch?;
        if batch.num_rows() > 0 {
            output_rows = output_rows.saturating_add(batch.num_rows());
            batches.push(batch);
        }
    }
    if let Some(label) = profile_label {
        eprintln!(
            "[dodam:scan-profile] i64_bloom_filter {label}: elapsed={:.3} ms setup={:.3} ms read_loop={:.3} ms row_groups={}/{} projected_columns={} rows={} batches={} zero_batches={} bytes={} metadata={:.3} ms planning={:.3} ms parquet_next={:.3} ms parquet_next_avg={:.3} ms parquet_next_max={:.3} ms parquet_calls={} parquet_eof={} parquet_rows={} parquet_batches={} avg_batch_rows={:.1}",
            started.elapsed().as_secs_f64() * 1000.0,
            nanos_to_millis(setup_nanos),
            nanos_to_millis(read_loop_nanos),
            row_group_count,
            reader.row_groups_total(),
            reader.projected_columns(),
            output_rows,
            batches.len(),
            reader.zero_row_batches(),
            reader.compressed_bytes_scanned(),
            nanos_to_millis(reader.metadata_nanos()),
            nanos_to_millis(reader.planning_nanos()),
            nanos_to_millis(reader.next_nanos()),
            average_nanos_millis(reader.next_nanos(), reader.next_calls()),
            nanos_to_millis(reader.max_next_nanos()),
            reader.next_calls(),
            reader.eof_calls(),
            reader.output_rows(),
            reader.output_batches(),
            average_rows_per_batch(reader.output_rows(), reader.output_batches()),
        );
    }
    Ok(batches)
}

#[allow(clippy::too_many_arguments)]
fn scan_dictionary_column_row_groups(
    path: PathBuf,
    batch_size: usize,
    projection: &Projection,
    row_groups: Vec<usize>,
    dictionary_columns: &[String],
    metadata_cache: &ParquetMetadataCache,
    file_cache: Arc<ParquetFileCache>,
    object_store: &dyn ObjectStore,
) -> Result<Vec<RecordBatch>> {
    let profile_label =
        scan_profile_enabled().then(|| parquet_map_profile_label(&path, projection));
    let row_group_count = row_groups.len();
    let started = Instant::now();
    let setup_started = Instant::now();
    let mut reader = ParquetBatchReader::try_new_with_row_groups_dictionary_columns(
        path,
        batch_size,
        projection,
        row_groups,
        dictionary_columns,
        metadata_cache,
        file_cache,
        object_store,
    )?;
    let setup_nanos = elapsed_nanos(setup_started.elapsed());
    let mut batches = Vec::new();
    let mut read_loop_nanos = 0_u64;
    let mut output_rows = 0_usize;
    loop {
        let next_started = Instant::now();
        let next = reader.next();
        read_loop_nanos = read_loop_nanos.saturating_add(elapsed_nanos(next_started.elapsed()));
        let Some(batch) = next else {
            break;
        };
        let batch = batch?;
        if batch.num_rows() > 0 {
            output_rows = output_rows.saturating_add(batch.num_rows());
            batches.push(batch);
        }
    }
    if let Some(label) = profile_label {
        eprintln!(
            "[dodam:scan-profile] dictionary_columns {label}: elapsed={:.3} ms setup={:.3} ms read_loop={:.3} ms row_groups={}/{} projected_columns={} rows={} batches={} zero_batches={} bytes={} metadata={:.3} ms planning={:.3} ms parquet_next={:.3} ms parquet_next_avg={:.3} ms parquet_next_max={:.3} ms parquet_calls={} parquet_eof={} parquet_rows={} parquet_batches={} avg_batch_rows={:.1}",
            started.elapsed().as_secs_f64() * 1000.0,
            nanos_to_millis(setup_nanos),
            nanos_to_millis(read_loop_nanos),
            row_group_count,
            reader.row_groups_total(),
            reader.projected_columns(),
            output_rows,
            batches.len(),
            reader.zero_row_batches(),
            reader.compressed_bytes_scanned(),
            nanos_to_millis(reader.metadata_nanos()),
            nanos_to_millis(reader.planning_nanos()),
            nanos_to_millis(reader.next_nanos()),
            average_nanos_millis(reader.next_nanos(), reader.next_calls()),
            nanos_to_millis(reader.max_next_nanos()),
            reader.next_calls(),
            reader.eof_calls(),
            reader.output_rows(),
            reader.output_batches(),
            average_rows_per_batch(reader.output_rows(), reader.output_batches()),
        );
    }
    Ok(batches)
}

#[allow(clippy::too_many_arguments)]
fn ordered_i64_decimal_group_sum_chunk(
    path: PathBuf,
    batch_size: usize,
    projection: &Projection,
    row_groups: Vec<usize>,
    key_column: &str,
    value_column: &str,
    threshold: f64,
    metadata_cache: &ParquetMetadataCache,
    file_cache: Arc<ParquetFileCache>,
    object_store: &dyn ObjectStore,
) -> Result<Option<OrderedGroupSumPartial>> {
    let mut reader = ParquetBatchReader::try_new_with_row_groups(
        path,
        batch_size,
        projection,
        row_groups,
        metadata_cache,
        file_cache,
        object_store,
    )?;
    let mut partial = OrderedGroupSumPartial::new();
    let mut current_key = None;
    let mut current_sum = 0.0;
    while let Some(batch) = reader.next() {
        let batch = batch?;
        let key_index = physical_batch_column_index(&batch, key_column)?;
        let value_index = physical_batch_column_index(&batch, value_column)?;
        let Some(keys) = batch
            .column(key_index)
            .as_any()
            .downcast_ref::<Int64Array>()
        else {
            return Ok(None);
        };
        let Some(values) = batch
            .column(value_index)
            .as_any()
            .downcast_ref::<Decimal128Array>()
        else {
            return Ok(None);
        };
        let scale = match values.data_type() {
            arrow::datatypes::DataType::Decimal128(_, scale) => 10_f64.powi(i32::from(*scale)),
            _ => return Ok(None),
        };
        if keys.null_count() == 0 && values.null_count() == 0 {
            for (&key, &value) in keys.values().iter().zip(values.values()) {
                let quantity = value as f64 / scale;
                if let Some(current) = current_key {
                    if key < current {
                        return Ok(None);
                    }
                    if key == current {
                        current_sum += quantity;
                        continue;
                    }
                    partial.push_run(current, current_sum, threshold);
                }
                current_key = Some(key);
                current_sum = quantity;
            }
            continue;
        }
        for row in 0..batch.num_rows() {
            if keys.is_null(row) || values.is_null(row) {
                continue;
            }
            let key = keys.value(row);
            let quantity = values.value(row) as f64 / scale;
            if let Some(current) = current_key {
                if key < current {
                    return Ok(None);
                }
                if key == current {
                    current_sum += quantity;
                    continue;
                }
                partial.push_run(current, current_sum, threshold);
            }
            current_key = Some(key);
            current_sum = quantity;
        }
    }
    partial.finish_current(current_key, current_sum, threshold);
    Ok(Some(partial))
}

fn merge_ordered_group_sum_partials(
    partials: Vec<Option<OrderedGroupSumPartial>>,
    threshold: f64,
) -> Result<Option<HashMap<i64, f64>>> {
    let mut chunks = Vec::with_capacity(partials.len());
    for partial in partials {
        let Some(partial) = partial else {
            return Ok(None);
        };
        if let (Some(first), Some(last)) = (&partial.first, &partial.last)
            && last.key < first.key
        {
            return Ok(None);
        }
        chunks.push(OrderedRowGroupChunk {
            output: partial.middle,
            first: partial.first,
            last: partial.last,
        });
    }
    let mut output = HashMap::new();
    merge_ordered_row_group_chunks(
        chunks,
        &mut output,
        |output, partial| output.extend(partial),
        |left, right| *left += right,
        |output, boundary| {
            if boundary.state > threshold {
                output.insert(boundary.key, boundary.state);
            }
        },
    );
    Ok(Some(output))
}

fn hash_physical_row(row: &impl Hash) -> usize {
    let mut hasher = DefaultHasher::new();
    row.hash(&mut hasher);
    hasher.finish() as usize
}

fn scan_projection_with_sort(
    projection: &Projection,
    filter: Option<&FilterExpr>,
    order_by: Option<&SortKey>,
) -> Projection {
    let mut projection = scan_projection(projection, filter);
    if let Some(order_by) = order_by {
        match &mut projection {
            Projection::All => {}
            Projection::Columns(columns) => {
                for sort in &order_by.expressions {
                    if !columns.iter().any(|column| column == &sort.column) {
                        columns.push(sort.column.clone());
                    }
                }
            }
        }
    }
    projection
}

fn scan_operators(
    limit: Option<usize>,
    distinct: bool,
    has_filter: bool,
    has_order_by: bool,
) -> Vec<ScanOperator> {
    let mut operators = Vec::new();
    if limit.is_some() {
        operators.push(ScanOperator::Limit);
    }
    if distinct {
        if has_order_by {
            operators.push(ScanOperator::Sort);
        }
        operators.push(ScanOperator::Distinct);
        operators.push(ScanOperator::Projection);
    } else {
        operators.push(ScanOperator::Projection);
        if has_order_by {
            operators.push(ScanOperator::Sort);
        }
    }
    if has_filter {
        operators.push(ScanOperator::Filter);
    }
    operators.push(ScanOperator::Scan);
    operators
}

fn projection_display(projection: &Projection) -> String {
    match projection {
        Projection::All => "*".to_string(),
        Projection::Columns(columns) => format!("[{}]", columns.join(",")),
    }
}

fn sort_key_display(order_by: &SortKey) -> String {
    order_by
        .expressions
        .iter()
        .map(|sort| {
            if sort.descending {
                format!("{} desc", sort.column)
            } else {
                format!("{} asc", sort.column)
            }
        })
        .collect::<Vec<_>>()
        .join(",")
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::Int32Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;

    use super::LocalShuffleStore;

    #[test]
    fn local_shuffle_write_rolls_files_by_target_bytes() {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let batches = vec![
            RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
            )
            .expect("first batch"),
            RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![4, 5, 6]))])
                .expect("second batch"),
        ];
        let mut store = LocalShuffleStore::new_with_file_target_bytes(1).expect("shuffle store");

        let metrics = store
            .write_partition(7, 0, &batches)
            .expect("write shuffle partition");
        let files = store.partition_files(7, 0).expect("partition files");

        assert_eq!(metrics.files, 2);
        assert_eq!(metrics.batches, 2);
        assert_eq!(metrics.rows, 6);
        assert!(metrics.bytes > 0);
        assert_eq!(files.len(), 2);
    }
}
