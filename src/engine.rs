use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use arrow::array::{Array, UInt32Array};
use arrow::ipc::writer::FileWriter as IpcFileWriter;
use arrow::record_batch::RecordBatch;
use arrow_row::{RowConverter, SortField};
use arrow_select::take::take_record_batch;

use crate::catalog::{
    FileFragmentStatistics, LocalParquetTable, PersistentCatalog, StorageFormat, TableProvider,
    TableScanSource, TableStatistics,
};
use crate::cost::{JoinCostInput, choose_join_strategy};
use crate::error::{DodamError, Result};
use crate::execution::{
    AggregateExpr, AggregateMetrics, ComparisonOp, DistinctExec, Expr, FilterExec, FilterExpr,
    HashJoinExec, IpcExec, JoinBuildSide, JoinType, LimitExec, LiteralValue, MemoryExec,
    PartitionedHashJoinExec, PartitionedHashJoinOptions, PhysicalPlan, PredicateSet, Projection,
    ProjectionExec, RecordBatchSink, ScanExec, ScanMetrics, ScanPlanMetrics, SendableBatchStream,
    SortExec, SortExpr, SortKey, SortMergeJoinExec, collect_aggregates, collect_grouped_aggregates,
    collect_metrics, scan_projection, write_stream_to_sink,
};
use crate::plan::{
    ExchangeKind, ExecutionGraphPlan, LogicalPlan, LogicalScan, PhysicalExecutionConfig,
    PhysicalJoinStrategy, PhysicalOperator, PhysicalPlanNode, PlanTableSource, TaskInput, TaskPlan,
};
use crate::storage::{
    LocalFileSystemObjectStore, ObjectStore, ParquetMetadataCache, plan_parquet_scan_tasks,
    read_parquet_file_statistics,
};

const LOCAL_SHUFFLE_FILE_TARGET_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Clone)]
pub struct DodamEngine {
    metadata_cache: Arc<ParquetMetadataCache>,
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
    pub has_filter: bool,
    pub distinct: bool,
    pub order_by: Option<SortKey>,
    pub estimated_bytes: u64,
    pub operators: Vec<ScanOperator>,
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
            has_filter: filter.is_some(),
            distinct: options.distinct,
            order_by: options.order_by,
            estimated_bytes,
            operators,
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
                self.metadata_cache.clone(),
                self.object_store.clone(),
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
        let source = self.plan_table_source(path).await?;
        self.aggregate_table(source, batch_size, aggregates, group_by, filter)
    }

    pub fn plan_table_aggregate(
        &self,
        source: TableScanSource,
        batch_size: usize,
        aggregates: Vec<AggregateExpr>,
        group_by: Vec<String>,
        filter: Option<FilterExpr>,
    ) -> Result<AggregatePlan> {
        let plan = self.plan_table_scan(
            source,
            batch_size,
            None,
            aggregate_projection(&aggregates, &group_by),
            filter,
            None,
        )?;
        Ok(AggregatePlan {
            scan: plan,
            aggregates,
            group_by,
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
        let projection = aggregate_projection(&aggregates, &group_by);
        let scan = self
            .plan_parquet_scan(path, batch_size, None, projection, filter, None)
            .await?;
        Ok(AggregatePlan {
            scan,
            aggregates,
            group_by,
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
        self.build_physical_scan_plan(plan).execute()
    }

    fn build_physical_scan_plan(&self, plan: ScanPlan) -> Box<dyn PhysicalPlan> {
        let scan = ScanExec::new(
            plan.source.fragments,
            plan.batch_size,
            plan.scan_projection,
            plan.pushdown_predicates,
            self.metadata_cache.clone(),
            self.object_store.clone(),
        );
        let mut physical: Box<dyn PhysicalPlan> = Box::new(scan);

        if let Some(filter) = plan.residual_filter {
            physical = Box::new(FilterExec::new(physical, filter));
        }

        if plan.distinct {
            physical = Box::new(ProjectionExec::new(physical, plan.output_projection));
            physical = Box::new(DistinctExec::new(physical));

            if let Some(order_by) = plan.order_by {
                physical = Box::new(SortExec::new(physical, order_by, plan.limit));
            }
        } else {
            if let Some(order_by) = plan.order_by {
                physical = Box::new(SortExec::new(physical, order_by, plan.limit));
            }
            physical = Box::new(ProjectionExec::new(physical, plan.output_projection));
        }

        if let Some(limit) = plan.limit {
            physical = Box::new(LimitExec::new(physical, limit));
        }

        physical
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
