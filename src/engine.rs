use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::{Arc, mpsc};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use arrow::array::{Array, Date32Array, Decimal128Array, Int64Array, UInt32Array};
use arrow::datatypes::DataType;
use arrow::ipc::writer::FileWriter as IpcFileWriter;
use arrow::record_batch::RecordBatch;
use arrow_row::{RowConverter, SortField};
use arrow_select::take::take_record_batch;
use parquet::arrow::arrow_reader::{RowSelection, RowSelector};

use crate::catalog::{
    FileFragmentStatistics, LocalParquetTable, PersistentCatalog, StorageFormat, TableProvider,
    TableScanSource, TableStatistics,
};
use crate::cost::{JoinCostInput, choose_join_strategy};
use crate::dense::DenseI64BoolLookup;
use crate::error::{DodamError, Result};
use crate::execution::metrics::ScanPlanMetricsCounter;
use crate::execution::{
    AggregateExpr, AggregateMetrics, ComparisonOp, DistinctExec, Expr, FilterExec, FilterExpr,
    HashJoinExec, IpcExec, JoinBuildSide, JoinType, LimitExec, LiteralValue, MemoryExec,
    PartitionedHashJoinExec, PartitionedHashJoinOptions, PhysicalPlan, PredicateSet, Projection,
    ProjectionExec, RecordBatchSink, ScanExec, ScanMetrics, ScanPlanMetrics, SendableBatchStream,
    SortExec, SortExpr, SortKey, SortMergeJoinExec, can_merge_partial_aggregates,
    collect_aggregates, collect_grouped_aggregates, collect_metrics, evaluate_filter_mask,
    merge_partial_aggregate_metrics, scan_projection, write_stream_to_sink,
};
use crate::plan::{
    ExchangeKind, ExecutionGraphPlan, LogicalPlan, LogicalScan, PhysicalExecutionConfig,
    PhysicalJoinStrategy, PhysicalOperator, PhysicalPlanNode, PlanTableSource, TaskInput, TaskPlan,
};
use crate::storage::{
    I64BloomPredicate, LocalFileSystemObjectStore, ObjectStore, ParquetBatchReader,
    ParquetFileCache, ParquetFileCacheStats, ParquetMetadataCache, plan_parquet_scan_tasks,
    read_parquet_file_statistics, read_parquet_i64_column_constant, read_parquet_i64_column_max,
};

const LOCAL_SHUFFLE_FILE_TARGET_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Clone)]
pub struct DodamEngine {
    metadata_cache: Arc<ParquetMetadataCache>,
    file_cache: Arc<ParquetFileCache>,
    object_store: Arc<dyn ObjectStore>,
    catalog_root: PathBuf,
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
    rows: usize,
    batches: usize,
    zero_batches: usize,
    total_nanos: u64,
    read_nanos: u64,
    reader_next_nanos: u64,
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
        self.rows = self.rows.saturating_add(other.rows);
        self.batches = self.batches.saturating_add(other.batches);
        self.zero_batches = self.zero_batches.saturating_add(other.zero_batches);
        self.total_nanos = self.total_nanos.saturating_add(other.total_nanos);
        self.read_nanos = self.read_nanos.saturating_add(other.read_nanos);
        self.reader_next_nanos = self
            .reader_next_nanos
            .saturating_add(other.reader_next_nanos);
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
        PhysicalPlanNode::new("AggregateExec")
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
            .child(self.scan.to_plan_node())
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
        let source = self.plan_table_source(path.clone()).await?;
        if source.format != StorageFormat::Parquet || source.fragments.len() != 1 {
            return Ok(None);
        }
        let local_path = source.fragments[0].parquet_local_path()?.to_path_buf();
        let predicate_projection = Projection::Columns(vec![
            "l_shipdate".to_string(),
            "l_discount".to_string(),
            "l_quantity".to_string(),
        ]);
        let plan = plan_parquet_scan_tasks(
            &local_path,
            &predicate_projection,
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
            return Ok(Some((0.0, 0)));
        }
        let chunks = row_groups
            .chunks(q06_late_materialized_row_group_chunk())
            .map(|chunk| chunk.to_vec())
            .collect::<Vec<_>>();
        let (sender, receiver) = mpsc::channel();
        for (index, row_groups) in chunks.iter().cloned().enumerate() {
            let sender = sender.clone();
            let path = local_path.clone();
            let metadata_cache = self.metadata_cache.clone();
            let file_cache = self.file_cache.clone();
            let object_store = self.object_store.clone();
            rayon::spawn(move || {
                let result = q06_late_materialized_revenue_sum_chunk(
                    path,
                    batch_size,
                    row_groups,
                    start_days,
                    end_days,
                    discount_low,
                    discount_high,
                    quantity_limit,
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
            .collect::<Vec<Option<Option<LateMaterializedChunkResult<(f64, u64)>>>>>();
        for _ in 0..chunks.len() {
            let (index, result) = receiver.recv().map_err(|_| {
                DodamError::UnsupportedSql("Q06 late materialized worker stopped".to_string())
            })?;
            partials[index] = Some(result?);
        }
        let mut sum = 0.0;
        let mut count = 0_u64;
        let mut metrics = LateMaterializedMetrics::default();
        for partial in partials {
            let Some(Some(partial)) = partial else {
                return Ok(None);
            };
            let (partial_sum, partial_count) = partial.output;
            sum += partial_sum;
            count += partial_count;
            metrics.add(partial.metrics);
        }
        log_late_materialized_metrics("Q06", metrics, chunks.len());
        Ok(Some((sum, count)))
    }

    pub(crate) async fn q14_late_materialized_promo_revenue(
        &self,
        path: PathBuf,
        batch_size: usize,
        start_days: i32,
        end_days: i32,
        promo_parts: Arc<DenseI64BoolLookup>,
    ) -> Result<Option<(f64, f64)>> {
        let source = self.plan_table_source(path.clone()).await?;
        if source.format != StorageFormat::Parquet || source.fragments.len() != 1 {
            return Ok(None);
        }
        let local_path = source.fragments[0].parquet_local_path()?.to_path_buf();
        let predicate_projection = Projection::Columns(vec!["l_shipdate".to_string()]);
        let plan = plan_parquet_scan_tasks(
            &local_path,
            &predicate_projection,
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
            return Ok(Some((0.0, 0.0)));
        }
        let chunks = row_groups
            .chunks(q14_late_materialized_row_group_chunk())
            .map(|chunk| chunk.to_vec())
            .collect::<Vec<_>>();
        let (sender, receiver) = mpsc::channel();
        for (index, row_groups) in chunks.iter().cloned().enumerate() {
            let sender = sender.clone();
            let path = local_path.clone();
            let promo_parts = promo_parts.clone();
            let metadata_cache = self.metadata_cache.clone();
            let file_cache = self.file_cache.clone();
            let object_store = self.object_store.clone();
            rayon::spawn(move || {
                let result = q14_late_materialized_promo_revenue_chunk(
                    path,
                    batch_size,
                    row_groups,
                    start_days,
                    end_days,
                    promo_parts,
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
            .collect::<Vec<Option<Option<LateMaterializedChunkResult<(f64, f64)>>>>>();
        for _ in 0..chunks.len() {
            let (index, result) = receiver.recv().map_err(|_| {
                DodamError::UnsupportedSql("Q14 late materialized worker stopped".to_string())
            })?;
            partials[index] = Some(result?);
        }
        let mut promo = 0.0;
        let mut total = 0.0;
        let mut metrics = LateMaterializedMetrics::default();
        for partial in partials {
            let Some(Some(partial)) = partial else {
                return Ok(None);
            };
            let (partial_promo, partial_total) = partial.output;
            promo += partial_promo;
            total += partial_total;
            metrics.add(partial.metrics);
        }
        log_late_materialized_metrics("Q14", metrics, chunks.len());
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
            let Some(sample_metrics) = sample_late_materialized_selection(
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
            if !policy.accepts(&sample_metrics) {
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
            let metadata_cache = self.metadata_cache.clone();
            let file_cache = self.file_cache.clone();
            let object_store = self.object_store.clone();
            let build_state = build_state.clone();
            let build_selection = build_selection.clone();
            let consume_payload = consume_payload.clone();
            let finish = finish.clone();
            rayon::spawn(move || {
                let result = late_materialized_chunk(
                    path,
                    batch_size,
                    row_groups,
                    &predicate_projection,
                    &payload_projection,
                    &metadata_cache,
                    file_cache,
                    object_store.as_ref(),
                    policy,
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
                DodamError::UnsupportedSql("late materialized worker stopped".to_string())
            })?;
            outputs[index] = Some(result?);
        }
        let mut results = Vec::with_capacity(outputs.len());
        for output in outputs {
            let Some(output) = output else {
                return Err(DodamError::UnsupportedSql(
                    "late materialized worker result missing".to_string(),
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
        let scan_projection = if options.distinct {
            scan_projection(&options.projection, filter.as_ref())
        } else {
            scan_projection_with_sort(
                &options.projection,
                filter.as_ref(),
                options.order_by.as_ref(),
            )
        };
        let predicates = PredicateSet::new(filter.clone());
        let estimated_bytes =
            self.estimate_scan_source_bytes(&source, &scan_projection, filter.as_ref())?;
        let pushdown_predicates = predicates.pushdown().to_vec();
        let residual_filter = predicates.residual().cloned();
        let operators = scan_operators(
            options.limit,
            options.distinct,
            filter.is_some(),
            options.order_by.is_some(),
        );
        Ok(ScanPlan {
            source,
            batch_size: options.batch_size,
            limit: options.limit,
            output_projection: options.projection,
            scan_projection,
            filter: filter.clone(),
            residual_filter,
            pushdown_predicates,
            row_filter_predicates: Vec::new(),
            has_filter: filter.is_some(),
            distinct: options.distinct,
            order_by: options.order_by,
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
            return self
                .try_late_materialized_parquet_aggregate(
                    path, batch_size, aggregates, group_by, filter,
                )
                .await;
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
        if partials.is_empty() {
            return Ok(None);
        }
        merge_partial_aggregate_metrics(partials, 1, &group_by, &aggregates).map(Some)
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
        let Some(partials) = self
            .late_materialized_parquet_map_with_policy(
                path,
                batch_size,
                column_read_plan.predicate_projection.clone(),
                column_read_plan.payload_projection.clone(),
                late_materialized_aggregate_row_group_chunk(),
                LateMaterializationPolicy::selective_with_selector_run_ratio(
                    late_materialized_aggregate_max_selected_ratio(),
                    late_materialized_aggregate_max_selector_run_ratio(),
                ),
                Vec::<RecordBatch>::new,
                {
                    let filter = filter.clone();
                    move |batch, selection, _batches| {
                        let mask = evaluate_filter_mask(&batch, &filter)?;
                        for row in 0..mask.len() {
                            selection.push(mask.is_valid(row) && mask.value(row));
                        }
                        Ok(Some(()))
                    }
                },
                |batch, batches| {
                    batches.push(batch);
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
        let scan = self
            .plan_parquet_scan(
                path,
                batch_size,
                None,
                column_read_plan.payload_projection.clone(),
                filter,
                None,
            )
            .await?;
        log_aggregate_column_read_plan(&column_read_plan);
        Ok(AggregatePlan {
            scan,
            aggregates,
            group_by,
            column_read_plan,
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
        let fragment_count = plan.scan.source.fragments.len();
        let stream = self.execute_scan_plan(plan.scan)?;
        if plan.group_by.is_empty() {
            collect_aggregates(stream, fragment_count, &plan.aggregates)
        } else {
            collect_grouped_aggregates(stream, fragment_count, &plan.group_by, &plan.aggregates)
        }
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
    std::env::var("DODAM_FUSED_PARQUET_AGG_ROW_GROUP_CHUNK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(4)
}

fn late_materialized_aggregate_enabled() -> bool {
    std::env::var("DODAM_ENABLE_LATE_MATERIALIZED_AGGREGATE")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

fn late_materialized_aggregate_row_group_chunk() -> usize {
    std::env::var("DODAM_LATE_AGG_ROW_GROUP_CHUNK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(4)
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
}

impl LateMaterializedMetrics {
    pub fn add(&mut self, other: Self) {
        self.total_rows = self.total_rows.saturating_add(other.total_rows);
        self.selected_rows = self.selected_rows.saturating_add(other.selected_rows);
        self.selector_runs = self.selector_runs.saturating_add(other.selector_runs);
    }
}

#[derive(Clone, Copy)]
pub struct LateMaterializationPolicy {
    max_selected_ratio: Option<f64>,
    max_selector_run_ratio: Option<f64>,
    max_selector_runs_per_selected: Option<f64>,
}

impl LateMaterializationPolicy {
    pub fn always() -> Self {
        Self {
            max_selected_ratio: None,
            max_selector_run_ratio: None,
            max_selector_runs_per_selected: None,
        }
    }

    pub fn selective(max_selected_ratio: f64) -> Self {
        Self {
            max_selected_ratio: Some(max_selected_ratio.clamp(0.0, 1.0)),
            max_selector_run_ratio: None,
            max_selector_runs_per_selected: None,
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
        }
    }

    pub fn with_selector_runs_per_selected(mut self, max_selector_runs_per_selected: f64) -> Self {
        self.max_selector_runs_per_selected = max_selector_runs_per_selected
            .is_finite()
            .then_some(max_selector_runs_per_selected.max(0.0));
        self
    }

    fn accepts(&self, metrics: &LateMaterializedMetrics) -> bool {
        if metrics.total_rows == 0 {
            return true;
        }
        if let Some(max_selected_ratio) = self.max_selected_ratio {
            let selected_ratio = metrics.selected_rows as f64 / metrics.total_rows as f64;
            if selected_ratio > max_selected_ratio {
                return false;
            }
        }
        if let Some(max_selector_run_ratio) = self.max_selector_run_ratio {
            let selector_run_ratio = metrics.selector_runs as f64 / metrics.total_rows as f64;
            if selector_run_ratio > max_selector_run_ratio {
                return false;
            }
        }
        if let Some(max_selector_runs_per_selected) = self.max_selector_runs_per_selected {
            if metrics.selected_rows == 0 {
                return true;
            }
            let selector_runs_per_selected =
                metrics.selector_runs as f64 / metrics.selected_rows as f64;
            if selector_runs_per_selected > max_selector_runs_per_selected {
                return false;
            }
        }
        true
    }

    fn has_selectivity_gate(&self) -> bool {
        self.max_selected_ratio.is_some()
            || self.max_selector_run_ratio.is_some()
            || self.max_selector_runs_per_selected.is_some()
    }
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

    fn finish(mut self) -> (Option<RowSelection>, LateMaterializedMetrics) {
        finish_late_selector_run(&mut self.selectors, self.current_selected, self.run_len);
        let metrics = LateMaterializedMetrics {
            total_rows: self.total_rows,
            selected_rows: self.selected_rows,
            selector_runs: self.selectors.len(),
        };
        let selection = (self.selected_rows > 0).then(|| RowSelection::from(self.selectors));
        (selection, metrics)
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

fn q06_late_materialized_row_group_chunk() -> usize {
    std::env::var("DODAM_Q06_LATE_ROW_GROUP_CHUNK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(2)
}

fn q14_late_materialized_row_group_chunk() -> usize {
    std::env::var("DODAM_Q14_LATE_ROW_GROUP_CHUNK")
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
    payload_columns >= predicate_columns.saturating_mul(2).max(1)
}

#[allow(clippy::too_many_arguments)]
fn sample_late_materialized_selection<State, BuildSelection>(
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
    BuildSelection: FnMut(RecordBatch, &mut LateSelectionBuilder, &mut State) -> Result<Option<()>>,
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
    while let Some(batch) = predicate_reader.next() {
        let batch = batch?;
        if build_selection(batch, &mut selection_builder, &mut state)?.is_none() {
            return Ok(None);
        }
    }
    let (_, metrics) = selection_builder.finish();
    Ok(Some(metrics))
}

#[allow(clippy::too_many_arguments)]
fn late_materialized_chunk<State, Output, BuildSelection, ConsumePayload, Finish>(
    path: PathBuf,
    batch_size: usize,
    row_groups: Vec<usize>,
    predicate_projection: &Projection,
    payload_projection: &Projection,
    metadata_cache: &ParquetMetadataCache,
    file_cache: Arc<ParquetFileCache>,
    object_store: &dyn ObjectStore,
    policy: LateMaterializationPolicy,
    mut state: State,
    mut build_selection: BuildSelection,
    mut consume_payload: ConsumePayload,
    finish: Finish,
) -> Result<Option<LateMaterializedChunkResult<Output>>>
where
    BuildSelection: FnMut(RecordBatch, &mut LateSelectionBuilder, &mut State) -> Result<Option<()>>,
    ConsumePayload: FnMut(RecordBatch, &mut State) -> Result<Option<()>>,
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
    while let Some(batch) = predicate_reader.next() {
        let batch = batch?;
        if build_selection(batch, &mut selection_builder, &mut state)?.is_none() {
            return Ok(None);
        }
    }
    let (row_selection, metrics) = selection_builder.finish();
    if !policy.accepts(&metrics) {
        return Ok(None);
    }
    if let Some(row_selection) = row_selection {
        let mut payload_reader = ParquetBatchReader::try_new_with_row_groups_selection(
            path,
            batch_size,
            payload_projection,
            row_groups,
            row_selection,
            metadata_cache,
            file_cache,
            object_store,
        )?;
        while let Some(batch) = payload_reader.next() {
            let batch = batch?;
            if consume_payload(batch, &mut state)?.is_none() {
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
        rows,
        batches,
        zero_batches: reader.zero_row_batches(),
        total_nanos,
        read_nanos,
        reader_next_nanos: reader.next_nanos(),
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
            "[dodam:tpch-profile] row_group_map {label}: chunk={} row_groups={} projected_columns={} rows={} batches={} zero_batches={} total={:.3} ms setup={:.3} ms metadata={:.3} ms planning={:.3} ms read_next={:.3} ms reader_next={:.3} ms reader_next_avg={:.3} ms reader_next_max={:.3} ms reader_calls={} reader_eof={} avg_batch_rows={:.1} consume={:.3} ms compressed={}/{}",
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
        rows,
        batches,
        zero_batches: reader.zero_row_batches(),
        total_nanos,
        read_nanos,
        reader_next_nanos: reader.next_nanos(),
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
            "[dodam:tpch-profile] dictionary_row_group_map {label}: chunk={} row_groups={} projected_columns={} rows={} batches={} zero_batches={} total={:.3} ms setup={:.3} ms metadata={:.3} ms planning={:.3} ms read_next={:.3} ms reader_next={:.3} ms reader_next_avg={:.3} ms reader_next_max={:.3} ms reader_calls={} reader_eof={} avg_batch_rows={:.1} consume={:.3} ms compressed={}/{}",
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
    let ratio = if metrics.total_rows == 0 {
        0.0
    } else {
        metrics.selected_rows as f64 / metrics.total_rows as f64
    };
    eprintln!(
        "[dodam:tpch-profile] {label}: late_materialized rows={} selected={} ratio={:.6} selector_runs={} chunks={chunks}",
        metrics.total_rows, metrics.selected_rows, ratio, metrics.selector_runs
    );
}

fn log_parquet_map_summary(kind: &str, label: Option<&str>, metrics: ParquetMapChunkMetrics) {
    let Some(label) = label else {
        return;
    };
    eprintln!(
        "[dodam:scan-profile] {kind}_summary {label}: chunks={} row_groups={} rows={} batches={} zero_batches={} total_sum={:.3} ms read_next={:.3} ms reader_next={:.3} ms reader_next_avg={:.3} ms reader_next_max={:.3} ms reader_calls={} reader_eof={} avg_batch_rows={:.1} consume={:.3} ms compressed={}/{}",
        metrics.chunks,
        metrics.row_groups,
        metrics.rows,
        metrics.batches,
        metrics.zero_batches,
        nanos_to_millis(metrics.total_nanos),
        nanos_to_millis(metrics.read_nanos),
        nanos_to_millis(metrics.reader_next_nanos),
        average_nanos_millis(metrics.reader_next_nanos, metrics.reader_calls),
        nanos_to_millis(metrics.reader_max_next_nanos),
        metrics.reader_calls,
        metrics.reader_eof,
        average_rows_per_batch(metrics.rows, metrics.batches),
        nanos_to_millis(metrics.consume_nanos),
        metrics.compressed_bytes_scanned,
        metrics.compressed_bytes_total,
    );
}

#[allow(clippy::too_many_arguments)]
fn q14_late_materialized_promo_revenue_chunk(
    path: PathBuf,
    batch_size: usize,
    row_groups: Vec<usize>,
    start_days: i32,
    end_days: i32,
    promo_parts: Arc<DenseI64BoolLookup>,
    metadata_cache: &ParquetMetadataCache,
    file_cache: Arc<ParquetFileCache>,
    object_store: &dyn ObjectStore,
) -> Result<Option<LateMaterializedChunkResult<(f64, f64)>>> {
    let predicate_projection = Projection::Columns(vec!["l_shipdate".to_string()]);
    let payload_projection = Projection::Columns(vec![
        "l_partkey".to_string(),
        "l_extendedprice".to_string(),
        "l_discount".to_string(),
    ]);
    let state = Q14LateState {
        promo_parts,
        promo: 0.0,
        total: 0.0,
    };
    late_materialized_chunk(
        path,
        batch_size,
        row_groups,
        &predicate_projection,
        &payload_projection,
        metadata_cache,
        file_cache,
        object_store,
        LateMaterializationPolicy::always(),
        state,
        |batch, selection, _state| {
            q14_build_date_selection_batch(batch, start_days, end_days, selection)
        },
        q14_consume_payload_batch,
        |state, _metrics| Ok(Some((state.promo, state.total))),
    )
}

struct Q14LateState {
    promo_parts: Arc<DenseI64BoolLookup>,
    promo: f64,
    total: f64,
}

fn q14_build_date_selection_batch(
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

fn q14_consume_payload_batch(batch: RecordBatch, state: &mut Q14LateState) -> Result<Option<()>> {
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
    let (price_precision, price_decimal_scale) = decimal128_precision_scale(extendedprices)?;
    let (discount_precision, discount_decimal_scale) = decimal128_precision_scale(discounts)?;
    if price_precision > 18 || discount_precision > 18 {
        return Ok(None);
    }
    let price_scale = decimal_scale_i64(price_decimal_scale)?;
    let discount_scale = decimal_scale_i64(discount_decimal_scale)?;
    let revenue_scale = 1.0 / ((price_scale as f64) * (discount_scale as f64));
    for row in 0..batch.num_rows() {
        let Some(is_promo) = state.promo_parts.get(partkeys.value(row)) else {
            continue;
        };
        let value = ((extendedprices.values()[row] as i64)
            * (discount_scale - discounts.values()[row] as i64)) as f64
            * revenue_scale;
        if is_promo {
            state.promo += value;
        }
        state.total += value;
    }
    Ok(Some(()))
}

#[allow(clippy::too_many_arguments)]
fn q06_late_materialized_revenue_sum_chunk(
    path: PathBuf,
    batch_size: usize,
    row_groups: Vec<usize>,
    start_days: i32,
    end_days: i32,
    discount_low: f64,
    discount_high: f64,
    quantity_limit: f64,
    metadata_cache: &ParquetMetadataCache,
    file_cache: Arc<ParquetFileCache>,
    object_store: &dyn ObjectStore,
) -> Result<Option<LateMaterializedChunkResult<(f64, u64)>>> {
    let predicate_projection = Projection::Columns(vec![
        "l_shipdate".to_string(),
        "l_discount".to_string(),
        "l_quantity".to_string(),
    ]);
    let payload_projection = Projection::Columns(vec!["l_extendedprice".to_string()]);
    let state = Q06LateState {
        selected_discounts: Vec::new(),
        discount_scale: None,
        extendedprice_scale: None,
        discount_offset: 0,
        sum: 0.0,
    };
    late_materialized_chunk(
        path,
        batch_size,
        row_groups,
        &predicate_projection,
        &payload_projection,
        metadata_cache,
        file_cache,
        object_store,
        LateMaterializationPolicy::always(),
        state,
        |batch, selection, state| {
            q06_build_selection_batch(
                batch,
                start_days,
                end_days,
                discount_low,
                discount_high,
                quantity_limit,
                selection,
                state,
            )
        },
        q06_consume_payload_batch,
        |state, _metrics| {
            if state.discount_offset != state.selected_discounts.len() {
                return Err(DodamError::UnsupportedSql(
                    "Q06 row selection payload mismatch".to_string(),
                ));
            }
            Ok(Some((state.sum, state.selected_discounts.len() as u64)))
        },
    )
}

struct Q06LateState {
    selected_discounts: Vec<i64>,
    discount_scale: Option<i64>,
    extendedprice_scale: Option<i64>,
    discount_offset: usize,
    sum: f64,
}

#[allow(clippy::too_many_arguments)]
fn q06_build_selection_batch(
    batch: RecordBatch,
    start_days: i32,
    end_days: i32,
    discount_low: f64,
    discount_high: f64,
    quantity_limit: f64,
    selection: &mut LateSelectionBuilder,
    state: &mut Q06LateState,
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

fn q06_consume_payload_batch(batch: RecordBatch, state: &mut Q06LateState) -> Result<Option<()>> {
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
    let discount_scale = state
        .discount_scale
        .ok_or_else(|| DodamError::UnsupportedSql("Q06 missing discount scale".to_string()))?;
    let revenue_scale = 1.0 / ((price_scale as f64) * (discount_scale as f64));
    for &extendedprice in extendedprices.values() {
        let discount = *state
            .selected_discounts
            .get(state.discount_offset)
            .ok_or_else(|| {
                DodamError::UnsupportedSql("Q06 row selection payload mismatch".to_string())
            })?;
        state.sum += ((extendedprice as i64) * discount) as f64 * revenue_scale;
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
    match *current_selected {
        Some(current) if current == selected => *run_len += 1,
        Some(current) => {
            selectors.push(if current {
                RowSelector::select(*run_len)
            } else {
                RowSelector::skip(*run_len)
            });
            *current_selected = Some(selected);
            *run_len = 1;
        }
        None => {
            *current_selected = Some(selected);
            *run_len = 1;
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
