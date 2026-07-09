use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use arrow::array::{
    Array, ArrayRef, BooleanArray, BooleanBuilder, Date32Array, Date64Array, Decimal128Array,
    DictionaryArray, Float64Array, Float64Builder, Int32Array, Int32Builder, Int64Array,
    Int64Builder, Scalar, StringArray, StringBuilder, TimestampMicrosecondArray,
    TimestampMillisecondArray, TimestampNanosecondArray, TimestampSecondArray, UInt32Array,
    UInt32Builder, UInt64Array, UInt64Builder, new_null_array,
};
use arrow::compute::filter_record_batch;
use arrow::compute::kernels::boolean::{and_kleene, is_not_null, is_null, not, or_kleene};
use arrow::compute::kernels::cmp::{eq, gt, gt_eq, lt, lt_eq, neq};
use arrow::datatypes::{DataType, Field, Int32Type, Schema, TimeUnit};
use arrow::ipc::reader::FileReader as IpcFileReader;
use arrow::ipc::writer::FileWriter as IpcFileWriter;
use arrow::record_batch::RecordBatch;
use arrow_ord::sort::{SortColumn, SortOptions, lexsort_to_indices};
use arrow_row::{OwnedRow, RowConverter, SortField};
use arrow_select::concat::concat_batches;
use arrow_select::take::take_record_batch;
use memchr::memmem::Finder;

use crate::catalog::{FileFragment, StorageFormat};
use crate::error::{DodamError, Result};
use crate::execution::logical::{
    AggregateExpr, AggregateMetrics, ComparisonExpr, ComparisonOp, Expr, FilterExpr, PhysicalPlan,
    Projection, SortExpr, SortKey,
};
use crate::execution::metrics::{
    RecordBatchSink, ScanMetrics, ScanPlanMetrics, ScanPlanMetricsCounter, SendableBatchStream,
};
use crate::execution::{
    DecimalDateRangeFilter, SingleKeyCountSumMinMaxVectorState, SingleKeyCountSumVectorState,
    aggregate_metrics_to_batches, collect_aggregates, collect_grouped_aggregates,
};
use crate::hash::{FastHashMap as JoinKeyHashMap, FastHashSet as JoinKeyHashSet};
use crate::plan::DirectPrimitiveFoldMode;
use crate::storage::{
    DirectPrimitiveColumnSpec, DirectPrimitiveColumnType, ObjectStore, ParquetBatchReader,
    ParquetFileCache, ParquetMetadataCache, ParquetScanTask, plan_parquet_scan_tasks,
    scan_parquet_primitive_columns_with_store,
};
use crate::vector::BatchView;

pub struct ScanExec {
    fragments: Vec<FileFragment>,
    batch_size: usize,
    projection: Projection,
    pruning_predicates: Vec<Expr>,
    row_filter_predicates: Vec<Expr>,
    metadata_cache: Arc<ParquetMetadataCache>,
    file_cache: Arc<ParquetFileCache>,
    object_store: Arc<dyn ObjectStore>,
    preserve_order: bool,
}

pub struct MemoryExec {
    batches: Vec<RecordBatch>,
}

pub struct LocalFoldExec {
    input: Box<dyn PhysicalPlan>,
    _group_by: Vec<String>,
    _aggregates: Vec<AggregateExpr>,
}

pub struct FinalMergeExec {
    input: Box<dyn PhysicalPlan>,
    group_by: Vec<String>,
    aggregates: Vec<AggregateExpr>,
}

pub struct DirectPrimitiveFoldExec {
    path: PathBuf,
    batch_size: usize,
    row_groups: Vec<usize>,
    columns: Vec<(String, String)>,
    mode: DirectPrimitiveFoldMode,
    file_cache: Arc<ParquetFileCache>,
    object_store: Arc<dyn ObjectStore>,
}

impl MemoryExec {
    pub fn new(batches: Vec<RecordBatch>) -> Self {
        Self { batches }
    }
}

impl PhysicalPlan for MemoryExec {
    fn execute(self: Box<Self>) -> Result<SendableBatchStream> {
        Ok(SendableBatchStream::from_batches(self.batches))
    }
}

impl LocalFoldExec {
    pub fn new(
        input: Box<dyn PhysicalPlan>,
        group_by: Vec<String>,
        aggregates: Vec<AggregateExpr>,
    ) -> Self {
        Self {
            input,
            _group_by: group_by,
            _aggregates: aggregates,
        }
    }
}

impl PhysicalPlan for LocalFoldExec {
    fn execute(self: Box<Self>) -> Result<SendableBatchStream> {
        self.input.execute()
    }
}

impl FinalMergeExec {
    pub fn new(
        input: Box<dyn PhysicalPlan>,
        group_by: Vec<String>,
        aggregates: Vec<AggregateExpr>,
    ) -> Self {
        Self {
            input,
            group_by,
            aggregates,
        }
    }
}

impl PhysicalPlan for FinalMergeExec {
    fn execute(self: Box<Self>) -> Result<SendableBatchStream> {
        let stream = self.input.execute()?;
        let metrics = if self.group_by.is_empty() {
            collect_aggregates(stream, 1, &self.aggregates)?
        } else {
            collect_grouped_aggregates(stream, 1, &self.group_by, &self.aggregates)?
        };
        Ok(SendableBatchStream::from_batches(
            aggregate_metrics_to_batches(&metrics, &self.group_by, &self.aggregates)?,
        ))
    }
}

impl DirectPrimitiveFoldExec {
    pub fn new(
        path: PathBuf,
        batch_size: usize,
        row_groups: Vec<usize>,
        columns: Vec<(String, String)>,
        mode: DirectPrimitiveFoldMode,
        file_cache: Arc<ParquetFileCache>,
        object_store: Arc<dyn ObjectStore>,
    ) -> Self {
        Self {
            path,
            batch_size,
            row_groups,
            columns,
            mode,
            file_cache,
            object_store,
        }
    }
}

impl PhysicalPlan for DirectPrimitiveFoldExec {
    fn execute(self: Box<Self>) -> Result<SendableBatchStream> {
        let (group_by, aggregates) = self.output_shape();
        let metrics = self.execute_metrics()?;
        let batches = aggregate_metrics_to_batches(&metrics, &group_by, &aggregates)?;
        Ok(SendableBatchStream::from_batches(batches))
    }
}

impl DirectPrimitiveFoldExec {
    fn output_shape(&self) -> (Vec<String>, Vec<AggregateExpr>) {
        match &self.mode {
            DirectPrimitiveFoldMode::SingleKeyCountSum {
                group_by,
                count,
                sum,
                ..
            } => (vec![group_by.clone()], vec![count.clone(), sum.clone()]),
            DirectPrimitiveFoldMode::SingleKeyCountSumMinMax {
                group_by,
                aggregates,
                ..
            } => (vec![group_by.clone()], aggregates.clone()),
        }
    }

    pub(crate) fn execute_metrics(self) -> Result<AggregateMetrics> {
        self.try_execute_metrics()?.ok_or_else(|| {
            DodamError::UnsupportedSql(
                "DirectPrimitiveFoldExec primitive scan contract was not satisfied".to_string(),
            )
        })
    }

    pub(crate) fn try_execute_metrics(self) -> Result<Option<AggregateMetrics>> {
        let columns = self
            .columns
            .iter()
            .map(|(name, column_type)| {
                Ok((
                    name.clone(),
                    parse_direct_primitive_column_type(column_type)?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        match self.mode {
            DirectPrimitiveFoldMode::SingleKeyCountSum {
                key_type,
                count,
                sum,
                ..
            } => {
                let Some((state, scan_metrics)) = scan_direct_primitive_parallel_fold(
                    self.path,
                    self.batch_size,
                    self.row_groups,
                    columns,
                    self.file_cache,
                    self.object_store,
                    || match key_type.as_str() {
                        "i32" | "I32" | "int32" | "Int32" => Ok(
                            SingleKeyCountSumVectorState::new_i32(count.clone(), sum.clone()),
                        ),
                        "i64" | "I64" | "int64" | "Int64" => Ok(
                            SingleKeyCountSumVectorState::new_i64(count.clone(), sum.clone()),
                        ),
                        _ => Err(DodamError::UnsupportedSql(format!(
                            "unsupported DirectPrimitiveFoldExec count/sum key type: {key_type}"
                        ))),
                    },
                    |state, batch| match key_type.as_str() {
                        "i32" | "I32" | "int32" | "Int32" => state.consume_i32_i64_batch(batch),
                        "i64" | "I64" | "int64" | "Int64" => state.consume_i64_i64_batch(batch),
                        _ => unreachable!("key type checked before scan"),
                    },
                    |state, partial| state.merge(partial),
                )?
                else {
                    return Ok(None);
                };
                let mut metrics = state.finish();
                metrics.fragments = 1;
                metrics.batches = scan_metrics.batches;
                metrics.rows = scan_metrics.rows;
                Ok(Some(metrics))
            }
            DirectPrimitiveFoldMode::SingleKeyCountSumMinMax {
                key_type,
                aggregates,
                decimal_precision,
                decimal_scale,
                decimal_min,
                decimal_max,
                date_min,
                date_max,
                ..
            } => {
                let filter = DecimalDateRangeFilter {
                    decimal_min: decimal_min.map(i128::from),
                    decimal_max: decimal_max.map(i128::from),
                    date_min,
                    date_max,
                };
                let Some((state, scan_metrics)) = scan_direct_primitive_parallel_fold(
                    self.path,
                    self.batch_size,
                    self.row_groups,
                    columns,
                    self.file_cache,
                    self.object_store,
                    || match key_type.as_str() {
                        "i32" | "I32" | "int32" | "Int32" => {
                            Ok(SingleKeyCountSumMinMaxVectorState::new_i32(
                                aggregates.clone(),
                                decimal_precision,
                                decimal_scale,
                            ))
                        }
                        "i64" | "I64" | "int64" | "Int64" => {
                            Ok(SingleKeyCountSumMinMaxVectorState::new_i64(
                                aggregates.clone(),
                                decimal_precision,
                                decimal_scale,
                            ))
                        }
                        _ => Err(DodamError::UnsupportedSql(format!(
                            "unsupported DirectPrimitiveFoldExec count/sum/min/max key type: {key_type}"
                        ))),
                    },
                    |state, batch| match key_type.as_str() {
                        "i32" | "I32" | "int32" | "Int32" => {
                            state.consume_i32_i64_decimal_date_batch(batch, &filter)
                        }
                        "i64" | "I64" | "int64" | "Int64" => {
                            state.consume_i64_i64_decimal_date_batch(batch, &filter)
                        }
                        _ => unreachable!("key type checked before scan"),
                    },
                    |state, partial| state.merge(partial),
                )?
                else {
                    return Ok(None);
                };
                let mut metrics = state.finish();
                metrics.fragments = 1;
                metrics.batches = scan_metrics.batches;
                metrics.rows = scan_metrics.rows;
                Ok(Some(metrics))
            }
        }
    }
}

fn scan_direct_primitive_parallel_fold<S, Init, Consume, Merge>(
    path: PathBuf,
    batch_size: usize,
    row_groups: Vec<usize>,
    columns: Vec<(String, DirectPrimitiveColumnType)>,
    file_cache: Arc<ParquetFileCache>,
    object_store: Arc<dyn ObjectStore>,
    init: Init,
    consume: Consume,
    merge: Merge,
) -> Result<Option<(S, crate::storage::DirectPrimitiveColumnScanMetrics)>>
where
    S: Send,
    Init: Fn() -> Result<S> + Sync,
    Consume: for<'a> Fn(&mut S, BatchView<'a>) -> Result<()> + Sync,
    Merge: Fn(&mut S, S) -> Result<()> + Sync,
{
    let mut state = init()?;
    let mut scan_metrics = crate::storage::DirectPrimitiveColumnScanMetrics::default();
    if row_groups.is_empty() {
        return Ok(Some((state, scan_metrics)));
    }
    if row_groups.len() <= 1 {
        let specs = borrowed_direct_primitive_specs(&columns);
        let Some(metrics) = scan_parquet_primitive_columns_with_store(
            &path,
            batch_size,
            &row_groups,
            &specs,
            file_cache,
            object_store.as_ref(),
            |columns| consume(&mut state, BatchView::from_raw_columns(columns)),
        )?
        else {
            return Ok(None);
        };
        return Ok(Some((state, metrics)));
    }

    let workers = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(4)
        .min(row_groups.len());
    let chunk_size = row_groups.len().div_ceil(workers).max(1);
    let (sender, receiver) = mpsc::channel();
    std::thread::scope(|scope| {
        for row_group_chunk in row_groups.chunks(chunk_size) {
            let sender = sender.clone();
            let path = path.clone();
            let columns = columns.clone();
            let row_groups = row_group_chunk.to_vec();
            let file_cache = file_cache.clone();
            let object_store = object_store.clone();
            let init = &init;
            let consume = &consume;
            scope.spawn(move || {
                let result: Result<Option<(S, crate::storage::DirectPrimitiveColumnScanMetrics)>> =
                    (|| {
                        let mut state = init()?;
                        let specs = borrowed_direct_primitive_specs(&columns);
                        let metrics = scan_parquet_primitive_columns_with_store(
                            &path,
                            batch_size,
                            &row_groups,
                            &specs,
                            file_cache,
                            object_store.as_ref(),
                            |columns| consume(&mut state, BatchView::from_raw_columns(columns)),
                        )?;
                        Ok(metrics.map(|metrics| (state, metrics)))
                    })();
                let _ = sender.send(result);
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
    Ok(Some((state, scan_metrics)))
}

fn borrowed_direct_primitive_specs(
    columns: &[(String, DirectPrimitiveColumnType)],
) -> Vec<DirectPrimitiveColumnSpec<'_>> {
    columns
        .iter()
        .map(|(name, column_type)| DirectPrimitiveColumnSpec {
            name: name.as_str(),
            column_type: *column_type,
        })
        .collect()
}

fn parse_direct_primitive_column_type(input: &str) -> Result<DirectPrimitiveColumnType> {
    match input {
        "i64" | "I64" | "int64" | "Int64" => Ok(DirectPrimitiveColumnType::I64),
        "i32" | "I32" | "int32" | "Int32" => Ok(DirectPrimitiveColumnType::I32),
        "date32" | "Date32" => Ok(DirectPrimitiveColumnType::Date32),
        input if input.starts_with("decimal128_i64_raw:") => {
            let mut parts = input.split(':');
            let _kind = parts.next();
            let precision = parts
                .next()
                .and_then(|value| value.parse::<u8>().ok())
                .ok_or_else(|| {
                    DodamError::UnsupportedSql(format!(
                        "invalid direct primitive decimal precision: {input}"
                    ))
                })?;
            let scale = parts
                .next()
                .and_then(|value| value.parse::<i8>().ok())
                .ok_or_else(|| {
                    DodamError::UnsupportedSql(format!(
                        "invalid direct primitive decimal scale: {input}"
                    ))
                })?;
            if parts.next().is_some() {
                return Err(DodamError::UnsupportedSql(format!(
                    "invalid direct primitive decimal type: {input}"
                )));
            }
            Ok(DirectPrimitiveColumnType::Decimal128Int64Raw { precision, scale })
        }
        _ => Err(DodamError::UnsupportedSql(format!(
            "unsupported direct primitive column type: {input}"
        ))),
    }
}

pub struct IpcExec {
    files: Vec<PathBuf>,
}

impl IpcExec {
    pub fn new(files: Vec<PathBuf>) -> Self {
        Self { files }
    }
}

impl PhysicalPlan for IpcExec {
    fn execute(self: Box<Self>) -> Result<SendableBatchStream> {
        Ok(SendableBatchStream::new(
            Box::new(IpcSendableBatchStream {
                inner: IpcBatchStream::new(self.files),
            }),
            Arc::default(),
        ))
    }
}

struct IpcSendableBatchStream {
    inner: IpcBatchStream,
}

impl Iterator for IpcSendableBatchStream {
    type Item = Result<RecordBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next_batch().transpose()
    }
}

impl ScanExec {
    pub fn new(
        fragments: Vec<FileFragment>,
        batch_size: usize,
        projection: Projection,
        pruning_predicates: Vec<Expr>,
        row_filter_predicates: Vec<Expr>,
        metadata_cache: Arc<ParquetMetadataCache>,
        file_cache: Arc<ParquetFileCache>,
        object_store: Arc<dyn ObjectStore>,
        preserve_order: bool,
    ) -> Self {
        Self {
            fragments,
            batch_size,
            projection,
            pruning_predicates,
            row_filter_predicates,
            metadata_cache,
            file_cache,
            object_store,
            preserve_order,
        }
    }

    pub fn fragments(&self) -> usize {
        self.fragments.len()
    }
}

impl PhysicalPlan for ScanExec {
    fn execute(self: Box<Self>) -> Result<SendableBatchStream> {
        let format = scan_format(&self.fragments)?;
        match format {
            StorageFormat::Parquet => Box::new(ParquetScanExec::new(
                self.fragments,
                self.batch_size,
                self.projection,
                self.pruning_predicates,
                self.row_filter_predicates,
                self.metadata_cache,
                self.file_cache,
                self.object_store,
                self.preserve_order,
            ))
            .execute(),
            StorageFormat::Csv | StorageFormat::Json | StorageFormat::ArrowIpc => {
                Err(DodamError::UnsupportedStorageFormat(format!("{format:?}")))
            }
        }
    }
}

fn scan_format(fragments: &[FileFragment]) -> Result<StorageFormat> {
    let Some(first) = fragments.first() else {
        return Ok(StorageFormat::Parquet);
    };
    let format = first.format;
    if fragments.iter().any(|fragment| fragment.format != format) {
        return Err(DodamError::UnsupportedStorageFormat(
            "mixed table fragment formats".to_string(),
        ));
    }
    Ok(format)
}

struct ParquetScanExec {
    fragments: Vec<FileFragment>,
    batch_size: usize,
    projection: Projection,
    pruning_predicates: Vec<Expr>,
    row_filter_predicates: Vec<Expr>,
    metadata_cache: Arc<ParquetMetadataCache>,
    file_cache: Arc<ParquetFileCache>,
    object_store: Arc<dyn ObjectStore>,
    preserve_order: bool,
}

impl ParquetScanExec {
    fn new(
        fragments: Vec<FileFragment>,
        batch_size: usize,
        projection: Projection,
        pruning_predicates: Vec<Expr>,
        row_filter_predicates: Vec<Expr>,
        metadata_cache: Arc<ParquetMetadataCache>,
        file_cache: Arc<ParquetFileCache>,
        object_store: Arc<dyn ObjectStore>,
        preserve_order: bool,
    ) -> Self {
        Self {
            fragments,
            batch_size,
            projection,
            pruning_predicates,
            row_filter_predicates,
            metadata_cache,
            file_cache,
            object_store,
            preserve_order,
        }
    }
}

impl PhysicalPlan for ParquetScanExec {
    fn execute(self: Box<Self>) -> Result<SendableBatchStream> {
        let metrics = Arc::new(ScanPlanMetricsCounter::default());
        let mut tasks = Vec::new();
        let mut row_groups_total = 0;
        let mut schema_columns_total = 0;
        let mut projected_columns_total = 0;
        let mut projected_columns_fixed_width = true;
        let mut compressed_bytes_total = 0;
        let mut compressed_bytes_scanned = 0;
        for fragment in &self.fragments {
            let parquet_projection =
                projection_without_partition_columns(&self.projection, &fragment.partition_values);
            let plan = plan_parquet_scan_tasks(
                fragment.parquet_local_path()?,
                &parquet_projection,
                &self.pruning_predicates,
                &self.metadata_cache,
                self.object_store.as_ref(),
            )?;
            row_groups_total += plan.row_groups_total;
            schema_columns_total += plan.schema_columns;
            projected_columns_total += plan.projected_columns;
            projected_columns_fixed_width &= plan.projected_columns_fixed_width;
            compressed_bytes_total += plan.compressed_bytes_total;
            compressed_bytes_scanned += plan.compressed_bytes_scanned;
            metrics.add_metadata_time(Duration::from_nanos(plan.metadata_nanos));
            metrics.add_planning_time(Duration::from_nanos(plan.planning_nanos));
            tasks.extend(plan.tasks.into_iter().map(|mut task| {
                task.partition_values = fragment.partition_values.clone();
                task
            }));
        }

        if tasks.is_empty() {
            return Ok(SendableBatchStream::new(
                Box::new(std::iter::empty()),
                metrics,
            ));
        }

        let pruned_columns = schema_columns_total.saturating_sub(projected_columns_total);
        let small_fixed_width_scan = pruned_columns == 0
            && compressed_bytes_scanned <= SEQUENTIAL_PARQUET_SCAN_BYTES
            && projected_columns_fixed_width
            && projected_columns_total <= SEQUENTIAL_PARQUET_SCAN_MAX_FIXED_WIDTH_COLUMNS
            && row_groups_total <= SEQUENTIAL_PARQUET_SCAN_MAX_FIXED_WIDTH_ROW_GROUPS;
        if self.preserve_order
            || tasks.len() == 1
            || (compressed_bytes_scanned <= SEQUENTIAL_PARQUET_SCAN_BYTES
                && pruned_columns >= SEQUENTIAL_PARQUET_SCAN_MIN_PRUNED_COLUMNS)
            || small_fixed_width_scan
        {
            return Ok(SendableBatchStream::new(
                Box::new(SequentialFragmentScanStream {
                    fragments: self.fragments,
                    batch_size: self.batch_size,
                    projection: self.projection,
                    pruning_predicates: self.pruning_predicates,
                    row_filter_predicates: self.row_filter_predicates,
                    current_reader: None,
                    current_partition_values: BTreeMap::new(),
                    next_fragment: 0,
                    decode_nanos: 0,
                    metrics: metrics.clone(),
                    metadata_cache: self.metadata_cache,
                    file_cache: self.file_cache,
                    object_store: self.object_store,
                }),
                metrics,
            ));
        }

        metrics.add_scan_plan(
            row_groups_total,
            tasks.len(),
            compressed_bytes_total,
            compressed_bytes_scanned,
        );
        Ok(SendableBatchStream::new(
            Box::new(ParallelParquetScanStream::new(
                tasks,
                self.batch_size,
                self.projection,
                self.metadata_cache,
                self.file_cache,
                self.object_store,
                self.row_filter_predicates,
                metrics.clone(),
            )),
            metrics,
        ))
    }
}

const SEQUENTIAL_PARQUET_SCAN_BYTES: u64 = 8 * 1024 * 1024;
const SEQUENTIAL_PARQUET_SCAN_MIN_PRUNED_COLUMNS: usize = 3;
const SEQUENTIAL_PARQUET_SCAN_MAX_FIXED_WIDTH_COLUMNS: usize = 2;
const SEQUENTIAL_PARQUET_SCAN_MAX_FIXED_WIDTH_ROW_GROUPS: usize = 16;

struct SequentialFragmentScanStream {
    fragments: Vec<FileFragment>,
    batch_size: usize,
    projection: Projection,
    pruning_predicates: Vec<Expr>,
    row_filter_predicates: Vec<Expr>,
    current_reader: Option<ParquetBatchReader>,
    current_partition_values: BTreeMap<String, String>,
    next_fragment: usize,
    decode_nanos: u64,
    metrics: Arc<ScanPlanMetricsCounter>,
    metadata_cache: Arc<ParquetMetadataCache>,
    file_cache: Arc<ParquetFileCache>,
    object_store: Arc<dyn ObjectStore>,
}

impl Iterator for SequentialFragmentScanStream {
    type Item = Result<RecordBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(reader) = &mut self.current_reader {
                let start = Instant::now();
                if let Some(batch) = reader.next() {
                    self.decode_nanos = self.decode_nanos.saturating_add(elapsed_nanos(start));
                    return Some(batch.and_then(|batch| {
                        add_partition_columns(
                            batch,
                            &self.projection,
                            &self.current_partition_values,
                        )
                    }));
                }
                self.metrics.add_parquet_reader_stats(
                    reader.next_calls(),
                    reader.eof_calls(),
                    reader.output_batches(),
                    reader.output_rows(),
                    reader.zero_row_batches(),
                    reader.next_nanos(),
                    reader.max_next_nanos(),
                );
                self.flush_decode_time();
                self.current_reader = None;
            }

            let fragment = self.fragments.get(self.next_fragment)?;
            self.next_fragment += 1;
            let path = match fragment.parquet_local_path() {
                Ok(path) => path,
                Err(error) => return Some(Err(error)),
            };
            let parquet_projection =
                projection_without_partition_columns(&self.projection, &fragment.partition_values);
            match ParquetBatchReader::try_new(
                path,
                self.batch_size,
                &parquet_projection,
                &self.pruning_predicates,
                &self.row_filter_predicates,
                &self.metadata_cache,
                self.file_cache.clone(),
                self.object_store.as_ref(),
            ) {
                Ok(reader) => {
                    self.metrics.add_scan_plan(
                        reader.row_groups_total(),
                        reader.row_groups_scanned(),
                        reader.compressed_bytes_total(),
                        reader.compressed_bytes_scanned(),
                    );
                    self.metrics
                        .add_metadata_time(Duration::from_nanos(reader.metadata_nanos()));
                    self.metrics
                        .add_planning_time(Duration::from_nanos(reader.planning_nanos()));
                    self.current_reader = Some(reader);
                    self.current_partition_values = fragment.partition_values.clone();
                }
                Err(error) => return Some(Err(error)),
            }
        }
    }
}

impl SequentialFragmentScanStream {
    fn flush_decode_time(&mut self) {
        if self.decode_nanos == 0 {
            return;
        }

        self.metrics
            .add_decode_time(Duration::from_nanos(self.decode_nanos));
        self.decode_nanos = 0;
    }
}

impl Drop for SequentialFragmentScanStream {
    fn drop(&mut self) {
        self.flush_decode_time();
    }
}

struct ParallelParquetScanStream {
    receiver: mpsc::Receiver<Result<RecordBatch>>,
}

struct ParquetScanTaskChunk {
    path: PathBuf,
    row_groups: Vec<usize>,
    partition_values: BTreeMap<String, String>,
}

const DEFAULT_PARALLEL_PARQUET_SCAN_ROW_GROUP_CHUNK: usize = 4;

fn parallel_parquet_scan_row_group_chunk() -> usize {
    std::env::var("DODAM_PARQUET_SCAN_ROW_GROUP_CHUNK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_PARALLEL_PARQUET_SCAN_ROW_GROUP_CHUNK)
}

impl ParallelParquetScanStream {
    fn new(
        tasks: Vec<ParquetScanTask>,
        batch_size: usize,
        projection: Projection,
        metadata_cache: Arc<ParquetMetadataCache>,
        file_cache: Arc<ParquetFileCache>,
        object_store: Arc<dyn ObjectStore>,
        row_filter_predicates: Vec<Expr>,
        metrics: Arc<ScanPlanMetricsCounter>,
    ) -> Self {
        let (sender, receiver) = mpsc::channel();
        let row_group_chunk = parallel_parquet_scan_row_group_chunk();
        for task in chunk_parquet_scan_tasks(tasks, row_group_chunk) {
            let sender = sender.clone();
            let projection = projection.clone();
            let parquet_projection =
                projection_without_partition_columns(&projection, &task.partition_values);
            let partition_values = task.partition_values.clone();
            let metadata_cache = metadata_cache.clone();
            let file_cache = file_cache.clone();
            let object_store = object_store.clone();
            let metrics = metrics.clone();
            let row_filter_predicates = row_filter_predicates.clone();
            rayon::spawn(move || {
                let mut reader = match ParquetBatchReader::try_new_with_row_groups_filtered(
                    &task.path,
                    batch_size,
                    &parquet_projection,
                    task.row_groups,
                    &row_filter_predicates,
                    &metadata_cache,
                    file_cache.clone(),
                    object_store.as_ref(),
                ) {
                    Ok(reader) => reader,
                    Err(error) => {
                        let _ = sender.send(Err(error));
                        return;
                    }
                };
                metrics.add_metadata_time(Duration::from_nanos(reader.metadata_nanos()));
                metrics.add_planning_time(Duration::from_nanos(reader.planning_nanos()));
                let mut decode_nanos = 0_u64;
                loop {
                    let start = Instant::now();
                    let Some(batch) = reader.next() else {
                        break;
                    };
                    decode_nanos = decode_nanos.saturating_add(elapsed_nanos(start));
                    let batch = batch.and_then(|batch| {
                        add_partition_columns(batch, &projection, &partition_values)
                    });
                    match batch {
                        Ok(batch) => {
                            if sender.send(Ok(batch)).is_err() {
                                break;
                            }
                        }
                        Err(error) => {
                            let _ = sender.send(Err(error));
                            break;
                        }
                    }
                }
                if decode_nanos > 0 {
                    metrics.add_decode_time(Duration::from_nanos(decode_nanos));
                }
                metrics.add_parquet_reader_stats(
                    reader.next_calls(),
                    reader.eof_calls(),
                    reader.output_batches(),
                    reader.output_rows(),
                    reader.zero_row_batches(),
                    reader.next_nanos(),
                    reader.max_next_nanos(),
                );
            });
        }
        drop(sender);

        Self { receiver }
    }
}

fn chunk_parquet_scan_tasks(
    tasks: Vec<ParquetScanTask>,
    target_row_groups: usize,
) -> Vec<ParquetScanTaskChunk> {
    let target_row_groups = target_row_groups.max(1);
    let mut chunks: Vec<ParquetScanTaskChunk> = Vec::new();
    for task in tasks {
        if let Some(chunk) = chunks.last_mut()
            && chunk.path == task.path
            && chunk.partition_values == task.partition_values
            && chunk.row_groups.len() < target_row_groups
        {
            chunk.row_groups.push(task.row_group);
            continue;
        }
        chunks.push(ParquetScanTaskChunk {
            path: task.path,
            row_groups: vec![task.row_group],
            partition_values: task.partition_values,
        });
    }
    chunks
}

impl Iterator for ParallelParquetScanStream {
    type Item = Result<RecordBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        self.receiver.recv().ok()
    }
}

fn projection_without_partition_columns(
    projection: &Projection,
    partition_values: &BTreeMap<String, String>,
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

fn add_partition_columns(
    batch: RecordBatch,
    projection: &Projection,
    partition_values: &BTreeMap<String, String>,
) -> Result<RecordBatch> {
    if partition_values.is_empty() {
        return Ok(batch);
    }

    let mut fields = batch
        .schema()
        .fields()
        .iter()
        .map(|field| field.as_ref().clone())
        .collect::<Vec<_>>();
    let mut columns = batch.columns().to_vec();
    for (name, value) in partition_values {
        if batch.schema().index_of(name).is_ok() {
            continue;
        }
        fields.push(Field::new(name, DataType::Utf8, false));
        columns
            .push(Arc::new(StringArray::from(vec![value.as_str(); batch.num_rows()])) as ArrayRef);
    }
    let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)?;
    apply_projection(batch, projection)
}

pub struct FilterExec {
    input: Box<dyn PhysicalPlan>,
    filter: FilterExpr,
}

impl FilterExec {
    pub fn new(input: Box<dyn PhysicalPlan>, filter: FilterExpr) -> Self {
        Self { input, filter }
    }
}

impl PhysicalPlan for FilterExec {
    fn execute(self: Box<Self>) -> Result<SendableBatchStream> {
        let Self { input, filter } = *self;
        let input = input.execute()?;
        let (input, metrics) = input.into_parts();
        Ok(SendableBatchStream::new(
            Box::new(FilterStream {
                input,
                filter,
                filter_nanos: 0,
                metrics: metrics.clone(),
            }),
            metrics,
        ))
    }
}

struct FilterStream {
    input: Box<dyn Iterator<Item = Result<RecordBatch>> + Send>,
    filter: FilterExpr,
    filter_nanos: u64,
    metrics: Arc<ScanPlanMetricsCounter>,
}

impl Iterator for FilterStream {
    type Item = Result<RecordBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        for batch in &mut self.input {
            match batch {
                Ok(batch) => {
                    let start = Instant::now();
                    let filtered = filter_batch(batch, &self.filter);
                    self.filter_nanos = self.filter_nanos.saturating_add(elapsed_nanos(start));
                    match filtered {
                        Ok(batch) if batch.num_rows() == 0 => continue,
                        Ok(batch) => return Some(Ok(batch)),
                        Err(error) => return Some(Err(error)),
                    }
                }
                Err(error) => return Some(Err(error)),
            }
        }

        self.flush_filter_time();
        None
    }
}

impl FilterStream {
    fn flush_filter_time(&mut self) {
        if self.filter_nanos == 0 {
            return;
        }

        self.metrics
            .add_filter_time(Duration::from_nanos(self.filter_nanos));
        self.filter_nanos = 0;
    }
}

impl Drop for FilterStream {
    fn drop(&mut self) {
        self.flush_filter_time();
    }
}

pub struct ProjectionExec {
    input: Box<dyn PhysicalPlan>,
    projection: Projection,
}

impl ProjectionExec {
    pub fn new(input: Box<dyn PhysicalPlan>, projection: Projection) -> Self {
        Self { input, projection }
    }
}

impl PhysicalPlan for ProjectionExec {
    fn execute(self: Box<Self>) -> Result<SendableBatchStream> {
        let Self { input, projection } = *self;
        let input = input.execute()?;
        let (input, metrics) = input.into_parts();
        Ok(SendableBatchStream::new(
            Box::new(ProjectionStream {
                input,
                projection,
                projection_nanos: 0,
                metrics: metrics.clone(),
            }),
            metrics,
        ))
    }
}

struct ProjectionStream {
    input: Box<dyn Iterator<Item = Result<RecordBatch>> + Send>,
    projection: Projection,
    projection_nanos: u64,
    metrics: Arc<ScanPlanMetricsCounter>,
}

impl Iterator for ProjectionStream {
    type Item = Result<RecordBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        let Some(batch) = self.input.next() else {
            self.flush_projection_time();
            return None;
        };
        let batch = match batch {
            Ok(batch) => batch,
            Err(error) => return Some(Err(error)),
        };

        let start = Instant::now();
        let projected = apply_projection(batch, &self.projection);
        self.projection_nanos = self.projection_nanos.saturating_add(elapsed_nanos(start));
        Some(projected)
    }
}

impl ProjectionStream {
    fn flush_projection_time(&mut self) {
        if self.projection_nanos == 0 {
            return;
        }

        self.metrics
            .add_projection_time(Duration::from_nanos(self.projection_nanos));
        self.projection_nanos = 0;
    }
}

impl Drop for ProjectionStream {
    fn drop(&mut self) {
        self.flush_projection_time();
    }
}

pub struct SortExec {
    input: Box<dyn PhysicalPlan>,
    sort: SortKey,
    limit: Option<usize>,
}

impl SortExec {
    pub fn new(input: Box<dyn PhysicalPlan>, sort: SortKey, limit: Option<usize>) -> Self {
        Self { input, sort, limit }
    }
}

impl PhysicalPlan for SortExec {
    fn execute(self: Box<Self>) -> Result<SendableBatchStream> {
        let input = self.input.execute()?;
        let (input, metrics) = input.into_parts();
        Ok(SendableBatchStream::new(
            Box::new(SortStream {
                input,
                sort: self.sort,
                limit: self.limit,
                emitted: false,
            }),
            metrics,
        ))
    }
}

struct SortStream {
    input: Box<dyn Iterator<Item = Result<RecordBatch>> + Send>,
    sort: SortKey,
    limit: Option<usize>,
    emitted: bool,
}

impl Iterator for SortStream {
    type Item = Result<RecordBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.emitted {
            return None;
        }
        self.emitted = true;

        let mut batches = Vec::new();
        for batch in &mut self.input {
            match batch {
                Ok(batch) if batch.num_rows() == 0 => {}
                Ok(batch) => batches.push(batch),
                Err(error) => return Some(Err(error)),
            }
        }

        if batches.is_empty() {
            return None;
        }

        Some(sort_batches(&batches, &self.sort, self.limit))
    }
}

fn sort_batches(
    batches: &[RecordBatch],
    sort: &SortKey,
    limit: Option<usize>,
) -> Result<RecordBatch> {
    if topk_batch_prune_enabled()
        && let Some(limit) = limit
        && batches.len() > 1
    {
        let candidates = batches
            .iter()
            .filter(|batch| batch.num_rows() > 0)
            .map(|batch| sort_single_batch(batch, sort, Some(limit)))
            .collect::<Result<Vec<_>>>()?;
        if candidates.is_empty() {
            return Ok(RecordBatch::new_empty(batches[0].schema()));
        }
        let schema = candidates[0].schema();
        let batch = if candidates.len() == 1 {
            candidates[0].clone()
        } else {
            concat_batches(&schema, candidates.iter())?
        };
        return sort_single_batch(&batch, sort, Some(limit));
    }
    let schema = batches[0].schema();
    let batch = if batches.len() == 1 {
        batches[0].clone()
    } else {
        concat_batches(&schema, batches.iter())?
    };
    sort_single_batch(&batch, sort, limit)
}

fn topk_batch_prune_enabled() -> bool {
    std::env::var("DODAM_DISABLE_TOPK_BATCH_PRUNE")
        .map(|value| value != "1" && !value.eq_ignore_ascii_case("true"))
        .unwrap_or(true)
}

fn sort_single_batch(
    batch: &RecordBatch,
    sort: &SortKey,
    limit: Option<usize>,
) -> Result<RecordBatch> {
    let sort_columns = sort
        .expressions
        .iter()
        .map(|sort| {
            let column_index = column_index(&batch, &sort.column)?;
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
    Ok(take_record_batch(batch, &indices)?)
}

pub struct DistinctExec {
    input: Box<dyn PhysicalPlan>,
}

impl DistinctExec {
    pub fn new(input: Box<dyn PhysicalPlan>) -> Self {
        Self { input }
    }
}

impl PhysicalPlan for DistinctExec {
    fn execute(self: Box<Self>) -> Result<SendableBatchStream> {
        let input = self.input.execute()?;
        let (input, metrics) = input.into_parts();
        Ok(SendableBatchStream::new(
            Box::new(DistinctStream {
                input,
                emitted: false,
            }),
            metrics,
        ))
    }
}

struct DistinctStream {
    input: Box<dyn Iterator<Item = Result<RecordBatch>> + Send>,
    emitted: bool,
}

impl Iterator for DistinctStream {
    type Item = Result<RecordBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.emitted {
            return None;
        }
        self.emitted = true;

        let mut batches = Vec::new();
        for batch in &mut self.input {
            match batch {
                Ok(batch) if batch.num_rows() == 0 => {}
                Ok(batch) => batches.push(batch),
                Err(error) => return Some(Err(error)),
            }
        }

        if batches.is_empty() {
            return None;
        }

        Some(distinct_batches(&batches))
    }
}

fn distinct_batches(batches: &[RecordBatch]) -> Result<RecordBatch> {
    let schema = batches[0].schema();
    let batch = concat_batches(&schema, batches.iter())?;
    let sort_fields = batch
        .schema()
        .fields()
        .iter()
        .map(|field| SortField::new(field.data_type().clone()))
        .collect::<Vec<_>>();
    let converter = RowConverter::new(sort_fields)?;
    let rows = converter.convert_columns(batch.columns())?;
    let mut seen = HashSet::<OwnedRow>::new();
    let mut indices = Vec::new();

    for (index, row) in rows.iter().enumerate() {
        if seen.insert(row.owned()) {
            let index = u32::try_from(index).map_err(|_| {
                DodamError::UnsupportedSql(
                    "DISTINCT currently supports up to u32::MAX rows".to_string(),
                )
            })?;
            indices.push(index);
        }
    }

    let indices = UInt32Array::from(indices);
    Ok(take_record_batch(&batch, &indices)?)
}

pub struct HashJoinExec {
    left: Box<dyn PhysicalPlan>,
    right: Box<dyn PhysicalPlan>,
    left_keys: Vec<String>,
    right_keys: Vec<String>,
    left_prefix: String,
    right_prefix: String,
    build_side: JoinBuildSide,
    join_type: JoinType,
    output_projection: Projection,
}

impl HashJoinExec {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        left: Box<dyn PhysicalPlan>,
        right: Box<dyn PhysicalPlan>,
        left_keys: Vec<String>,
        right_keys: Vec<String>,
        left_prefix: String,
        right_prefix: String,
        build_side: JoinBuildSide,
        join_type: JoinType,
        output_projection: Projection,
    ) -> Self {
        Self {
            left,
            right,
            left_keys,
            right_keys,
            left_prefix,
            right_prefix,
            build_side,
            join_type,
            output_projection,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinBuildSide {
    Left,
    Right,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum JoinType {
    #[default]
    Inner,
    Left,
    Right,
    Full,
    Semi,
}

pub struct PartitionedHashJoinExec {
    left: Box<dyn PhysicalPlan>,
    right: Box<dyn PhysicalPlan>,
    left_keys: Vec<String>,
    right_keys: Vec<String>,
    left_prefix: String,
    right_prefix: String,
    partitions: usize,
    memory_limit_bytes: u64,
    join_type: JoinType,
    output_projection: Projection,
}

pub struct SortMergeJoinExec {
    left: Box<dyn PhysicalPlan>,
    right: Box<dyn PhysicalPlan>,
    left_key: String,
    right_key: String,
    left_prefix: String,
    right_prefix: String,
}

impl SortMergeJoinExec {
    pub fn new(
        left: Box<dyn PhysicalPlan>,
        right: Box<dyn PhysicalPlan>,
        left_key: String,
        right_key: String,
        left_prefix: String,
        right_prefix: String,
    ) -> Self {
        Self {
            left,
            right,
            left_key,
            right_key,
            left_prefix,
            right_prefix,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartitionedHashJoinOptions {
    pub partitions: usize,
    pub memory_limit_bytes: u64,
    pub join_type: JoinType,
    pub output_projection: Projection,
}

impl PartitionedHashJoinExec {
    pub fn new(
        left: Box<dyn PhysicalPlan>,
        right: Box<dyn PhysicalPlan>,
        left_keys: Vec<String>,
        right_keys: Vec<String>,
        left_prefix: String,
        right_prefix: String,
        options: PartitionedHashJoinOptions,
    ) -> Self {
        Self {
            left,
            right,
            left_keys,
            right_keys,
            left_prefix,
            right_prefix,
            partitions: options.partitions.max(1),
            memory_limit_bytes: options.memory_limit_bytes.max(1),
            join_type: options.join_type,
            output_projection: options.output_projection,
        }
    }
}

impl PhysicalPlan for PartitionedHashJoinExec {
    fn execute(self: Box<Self>) -> Result<SendableBatchStream> {
        let left = self.left.execute()?;
        let right = self.right.execute()?;
        let (left, metrics) = left.into_parts();
        let (right, _) = right.into_parts();
        let output_projection = JoinOutputProjection::from_projection(
            &self.output_projection,
            &self.left_prefix,
            &self.right_prefix,
            self.join_type,
        );
        Ok(SendableBatchStream::new(
            Box::new(PartitionedHashJoinStream {
                left,
                right,
                left_keys: self.left_keys,
                right_keys: self.right_keys,
                left_prefix: self.left_prefix,
                right_prefix: self.right_prefix,
                partitions: self.partitions,
                memory_limit_bytes: self.memory_limit_bytes,
                join_type: self.join_type,
                output_projection,
                spills: Vec::new(),
                pending: VecDeque::new(),
                active: None,
                pending_output: VecDeque::new(),
                metrics: metrics.clone(),
                initialized: false,
            }),
            metrics,
        ))
    }
}

impl PhysicalPlan for SortMergeJoinExec {
    fn execute(self: Box<Self>) -> Result<SendableBatchStream> {
        let left = self.left.execute()?;
        let right = self.right.execute()?;
        let (left, metrics) = left.into_parts();
        let (right, _) = right.into_parts();
        Ok(SendableBatchStream::new(
            Box::new(SortMergeJoinStream {
                left,
                right,
                left_key: self.left_key,
                right_key: self.right_key,
                left_prefix: self.left_prefix,
                right_prefix: self.right_prefix,
                metrics: metrics.clone(),
                emitted: false,
            }),
            metrics,
        ))
    }
}

impl PhysicalPlan for HashJoinExec {
    fn execute(self: Box<Self>) -> Result<SendableBatchStream> {
        let left = self.left.execute()?;
        let right = self.right.execute()?;
        let (left, metrics) = left.into_parts();
        let (right, _) = right.into_parts();
        let output_projection = JoinOutputProjection::from_projection(
            &self.output_projection,
            &self.left_prefix,
            &self.right_prefix,
            self.join_type,
        );
        Ok(SendableBatchStream::new(
            Box::new(HashJoinStream {
                left,
                right,
                left_keys: self.left_keys,
                right_keys: self.right_keys,
                left_prefix: self.left_prefix,
                right_prefix: self.right_prefix,
                build: None,
                build_side: self.build_side,
                join_type: self.join_type,
                output_projection,
                metrics: metrics.clone(),
                pending_output: VecDeque::new(),
                built: false,
                matched_build: None,
                emitted_unmatched_build: false,
                probe_template: None,
                profile_logged: false,
            }),
            metrics,
        ))
    }

    fn execute_to_sink(self: Box<Self>, sink: &mut dyn RecordBatchSink) -> Result<ScanPlanMetrics> {
        let left = self.left.execute()?;
        let right = self.right.execute()?;
        let (mut left, metrics) = left.into_parts();
        let (mut right, _) = right.into_parts();
        let output_projection = JoinOutputProjection::from_projection(
            &self.output_projection,
            &self.left_prefix,
            &self.right_prefix,
            self.join_type,
        );
        let mut probe_template = None;

        if sink.discards_output() && matches!(self.join_type, JoinType::Inner | JoinType::Semi) {
            let _ = (left, right);
            sink.finish()?;
            return Ok(metrics.snapshot());
        }

        let build = match self.build_side {
            JoinBuildSide::Left => {
                let left = collect_non_empty_batches(&mut left)?;
                if left.is_empty() {
                    sink.finish()?;
                    return Ok(metrics.snapshot());
                }
                let mode = if self.join_type == JoinType::Semi {
                    HashBuildMode::FastSemi
                } else {
                    HashBuildMode::FastSingleKey
                };
                let build_started = Instant::now();
                let build = build_hash_join_input(
                    &left,
                    &self.left_keys,
                    mode,
                    should_emit_unmatched_build(self.join_type, self.build_side),
                    &metrics,
                )?;
                metrics.add_join_build_time(build_started.elapsed());
                build
            }
            JoinBuildSide::Right => {
                let right = collect_non_empty_batches(&mut right)?;
                if right.is_empty() {
                    sink.finish()?;
                    return Ok(metrics.snapshot());
                }
                let mode = if self.join_type == JoinType::Semi {
                    HashBuildMode::FastSemi
                } else {
                    HashBuildMode::FastSingleKey
                };
                let build_started = Instant::now();
                let build = build_hash_join_input(
                    &right,
                    &self.right_keys,
                    mode,
                    should_emit_unmatched_build(self.join_type, self.build_side),
                    &metrics,
                )?;
                metrics.add_join_build_time(build_started.elapsed());
                build
            }
        };
        let mut matched_build = MatchedBuildTracker::for_build(&build);
        let discard_output =
            sink.discards_output() && matches!(self.join_type, JoinType::Inner | JoinType::Semi);

        match self.build_side {
            JoinBuildSide::Left => {
                for right in right.by_ref() {
                    let right = right?;
                    probe_template.get_or_insert_with(|| right.clone());
                    if right.num_rows() == 0 {
                        continue;
                    }
                    if discard_output {
                        discard_hash_join_batches(
                            &right,
                            &build,
                            &self.right_keys,
                            self.join_type,
                            None,
                            &metrics,
                        )?;
                        continue;
                    }
                    let batches = probe_hash_join_batches(
                        &right,
                        &build,
                        &self.right_keys,
                        &self.left_prefix,
                        &self.right_prefix,
                        self.build_side,
                        self.join_type,
                        should_track_matched_build(self.join_type, self.build_side)
                            .then_some(&mut matched_build),
                        None,
                        None,
                        &output_projection,
                        &metrics,
                    )?;
                    write_record_batches_to_sink(batches, sink)?;
                }
            }
            JoinBuildSide::Right => {
                for left in left.by_ref() {
                    let left = left?;
                    probe_template.get_or_insert_with(|| left.clone());
                    if left.num_rows() == 0 {
                        continue;
                    }
                    if discard_output {
                        discard_hash_join_batches(
                            &left,
                            &build,
                            &self.left_keys,
                            self.join_type,
                            None,
                            &metrics,
                        )?;
                        continue;
                    }
                    if try_probe_i32_semi_join_to_i32_sink(
                        &left,
                        &build,
                        &self.left_keys,
                        self.join_type,
                        &output_projection,
                        &metrics,
                        sink,
                    )? {
                        continue;
                    }
                    if try_probe_i32_dense_join_to_i32_utf8_sink(
                        &left,
                        &build,
                        &self.left_keys,
                        self.join_type,
                        &output_projection,
                        &metrics,
                        sink,
                    )? {
                        continue;
                    }
                    if try_probe_unique_join_to_i32_utf8_sink(
                        &left,
                        &build,
                        &self.left_keys,
                        self.join_type,
                        &output_projection,
                        &metrics,
                        sink,
                    )? {
                        continue;
                    }
                    let batches = probe_hash_join_batches(
                        &left,
                        &build,
                        &self.left_keys,
                        &self.left_prefix,
                        &self.right_prefix,
                        self.build_side,
                        self.join_type,
                        should_track_matched_build(self.join_type, self.build_side)
                            .then_some(&mut matched_build),
                        None,
                        None,
                        &output_projection,
                        &metrics,
                    )?;
                    write_record_batches_to_sink(batches, sink)?;
                }
            }
        }

        let mut emitted_unmatched_build = false;
        if !discard_output {
            let batches = emit_unmatched_build_if_needed(
                &build,
                Some(&matched_build),
                self.build_side,
                self.join_type,
                &self.left_prefix,
                &self.right_prefix,
                probe_template.as_ref(),
                &metrics,
                &mut emitted_unmatched_build,
            )?;
            write_record_batches_to_sink(batches, sink)?;
        }

        sink.finish()?;
        Ok(metrics.snapshot())
    }
}

fn write_record_batches_to_sink(
    batches: Vec<RecordBatch>,
    sink: &mut dyn RecordBatchSink,
) -> Result<()> {
    for batch in batches {
        sink.write_batch(&batch)?;
    }
    Ok(())
}

struct SortMergeJoinStream {
    left: Box<dyn Iterator<Item = Result<RecordBatch>> + Send>,
    right: Box<dyn Iterator<Item = Result<RecordBatch>> + Send>,
    left_key: String,
    right_key: String,
    left_prefix: String,
    right_prefix: String,
    metrics: Arc<ScanPlanMetricsCounter>,
    emitted: bool,
}

impl Iterator for SortMergeJoinStream {
    type Item = Result<RecordBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.emitted {
            return None;
        }
        self.emitted = true;

        let left = match collect_non_empty_batches(&mut self.left) {
            Ok(batches) => batches,
            Err(error) => return Some(Err(error)),
        };
        let right = match collect_non_empty_batches(&mut self.right) {
            Ok(batches) => batches,
            Err(error) => return Some(Err(error)),
        };
        if left.is_empty() || right.is_empty() {
            return None;
        }

        Some(sort_merge_join_batches(
            &left,
            &right,
            &self.left_key,
            &self.right_key,
            &self.left_prefix,
            &self.right_prefix,
            &self.metrics,
        ))
    }
}

struct HashJoinStream {
    left: Box<dyn Iterator<Item = Result<RecordBatch>> + Send>,
    right: Box<dyn Iterator<Item = Result<RecordBatch>> + Send>,
    left_keys: Vec<String>,
    right_keys: Vec<String>,
    left_prefix: String,
    right_prefix: String,
    build: Option<HashJoinBuild>,
    build_side: JoinBuildSide,
    join_type: JoinType,
    output_projection: JoinOutputProjection,
    metrics: Arc<ScanPlanMetricsCounter>,
    pending_output: VecDeque<RecordBatch>,
    built: bool,
    matched_build: Option<MatchedBuildTracker>,
    emitted_unmatched_build: bool,
    probe_template: Option<RecordBatch>,
    profile_logged: bool,
}

impl Iterator for HashJoinStream {
    type Item = Result<RecordBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(batch) = self.pending_output.pop_front() {
            return Some(Ok(batch));
        }
        if !self.built {
            self.built = true;
            let build = match self.build_side {
                JoinBuildSide::Left => {
                    let left = match collect_non_empty_batches(&mut self.left) {
                        Ok(batches) => batches,
                        Err(error) => return Some(Err(error)),
                    };
                    if left.is_empty() {
                        self.log_profile_once();
                        return None;
                    }
                    let mode = if self.join_type == JoinType::Semi {
                        HashBuildMode::FastSemi
                    } else {
                        HashBuildMode::FastSingleKey
                    };
                    build_hash_join_input(
                        &left,
                        &self.left_keys,
                        mode,
                        should_emit_unmatched_build(self.join_type, self.build_side),
                        &self.metrics,
                    )
                }
                JoinBuildSide::Right => {
                    let right = match collect_non_empty_batches(&mut self.right) {
                        Ok(batches) => batches,
                        Err(error) => return Some(Err(error)),
                    };
                    if right.is_empty() {
                        self.log_profile_once();
                        return None;
                    }
                    let mode = if self.join_type == JoinType::Semi {
                        HashBuildMode::FastSemi
                    } else {
                        HashBuildMode::FastSingleKey
                    };
                    build_hash_join_input(
                        &right,
                        &self.right_keys,
                        mode,
                        should_emit_unmatched_build(self.join_type, self.build_side),
                        &self.metrics,
                    )
                }
            };
            self.build = match build {
                Ok(build) => {
                    self.matched_build =
                        should_track_matched_build(self.join_type, self.build_side)
                            .then(|| MatchedBuildTracker::for_build(&build));
                    Some(build)
                }
                Err(error) => return Some(Err(error)),
            };
        }

        let build = self.build.as_ref()?;
        match self.build_side {
            JoinBuildSide::Left => {
                for right in self.right.by_ref() {
                    match right {
                        Ok(right) if right.num_rows() == 0 => {
                            self.probe_template.get_or_insert_with(|| right.clone());
                            continue;
                        }
                        Ok(right) => {
                            self.probe_template.get_or_insert_with(|| right.clone());
                            match probe_hash_join_batches(
                                &right,
                                build,
                                &self.right_keys,
                                &self.left_prefix,
                                &self.right_prefix,
                                self.build_side,
                                self.join_type,
                                self.matched_build.as_mut(),
                                None,
                                None,
                                &self.output_projection,
                                &self.metrics,
                            ) {
                                Ok(batches) if batches.is_empty() => continue,
                                Ok(mut batches) => {
                                    self.pending_output.extend(batches.drain(1..));
                                    return Some(Ok(batches.remove(0)));
                                }
                                Err(error) => return Some(Err(error)),
                            }
                        }
                        Err(error) => return Some(Err(error)),
                    }
                }
            }
            JoinBuildSide::Right => {
                for left in self.left.by_ref() {
                    match left {
                        Ok(left) if left.num_rows() == 0 => {
                            self.probe_template.get_or_insert_with(|| left.clone());
                            continue;
                        }
                        Ok(left) => {
                            self.probe_template.get_or_insert_with(|| left.clone());
                            match probe_hash_join_batches(
                                &left,
                                build,
                                &self.left_keys,
                                &self.left_prefix,
                                &self.right_prefix,
                                self.build_side,
                                self.join_type,
                                self.matched_build.as_mut(),
                                None,
                                None,
                                &self.output_projection,
                                &self.metrics,
                            ) {
                                Ok(batches) if batches.is_empty() => continue,
                                Ok(mut batches) => {
                                    self.pending_output.extend(batches.drain(1..));
                                    return Some(Ok(batches.remove(0)));
                                }
                                Err(error) => return Some(Err(error)),
                            }
                        }
                        Err(error) => return Some(Err(error)),
                    }
                }
            }
        }

        match emit_unmatched_build_if_needed(
            build,
            self.matched_build.as_ref(),
            self.build_side,
            self.join_type,
            &self.left_prefix,
            &self.right_prefix,
            self.probe_template.as_ref(),
            &self.metrics,
            &mut self.emitted_unmatched_build,
        ) {
            Ok(mut batches) if !batches.is_empty() => {
                self.pending_output.extend(batches.drain(1..));
                return Some(Ok(batches.remove(0)));
            }
            Ok(_) => {}
            Err(error) => return Some(Err(error)),
        }

        self.log_profile_once();
        None
    }
}

impl HashJoinStream {
    fn log_profile_once(&mut self) {
        if self.profile_logged || !join_profile_enabled() {
            return;
        }
        self.profile_logged = true;
        let metrics = self.metrics.snapshot();
        eprintln!(
            "[dodam:join-profile] build_rows={} probe_rows={} output_rows={} build={:.3}ms materialize={:.3}ms peak_build_bytes={} bloom_filtered={} spills={} spill_bytes={}",
            metrics.join_build_rows,
            metrics.join_probe_rows,
            metrics.join_output_rows,
            nanos_to_millis(metrics.join_build_nanos),
            nanos_to_millis(metrics.join_materialize_nanos),
            metrics.join_peak_build_bytes,
            metrics.join_bloom_filtered_rows,
            metrics.join_spill_files,
            metrics.join_spill_bytes,
        );
    }
}

fn join_profile_enabled() -> bool {
    std::env::var("DODAM_JOIN_PROFILE").is_ok_and(|value| {
        matches!(
            value.to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

fn nanos_to_millis(nanos: u64) -> f64 {
    nanos as f64 / 1_000_000.0
}

struct PartitionedHashJoinStream {
    left: Box<dyn Iterator<Item = Result<RecordBatch>> + Send>,
    right: Box<dyn Iterator<Item = Result<RecordBatch>> + Send>,
    left_keys: Vec<String>,
    right_keys: Vec<String>,
    left_prefix: String,
    right_prefix: String,
    partitions: usize,
    memory_limit_bytes: u64,
    join_type: JoinType,
    output_projection: JoinOutputProjection,
    spills: Vec<SpilledJoin>,
    pending: VecDeque<PartitionTask>,
    active: Option<PartitionJoinState>,
    pending_output: VecDeque<RecordBatch>,
    metrics: Arc<ScanPlanMetricsCounter>,
    initialized: bool,
}

impl Iterator for PartitionedHashJoinStream {
    type Item = Result<RecordBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(batch) = self.pending_output.pop_front() {
            return Some(Ok(batch));
        }
        if !self.initialized {
            self.initialized = true;
            let spill = match spill_join_inputs(
                &mut self.left,
                &mut self.right,
                &self.left_keys,
                &self.right_keys,
                self.partitions,
                0,
                &self.metrics,
            ) {
                Ok(spill) => spill,
                Err(error) => return Some(Err(error)),
            };
            self.spills.push(spill);
            self.pending
                .extend((0..self.partitions).map(|partition| PartitionTask {
                    spill_index: 0,
                    partition,
                    depth: 0,
                }));
        }

        loop {
            if let Some(active) = &mut self.active {
                loop {
                    match active.next_output_batches(
                        &self.left_prefix,
                        &self.right_prefix,
                        &self.output_projection,
                        &self.metrics,
                    ) {
                        Ok(Some(batches)) if batches.is_empty() => continue,
                        Ok(Some(mut batches)) => {
                            self.pending_output.extend(batches.drain(1..));
                            return Some(Ok(batches.remove(0)));
                        }
                        Ok(None) => break,
                        Err(error) => return Some(Err(error)),
                    }
                }
                self.active = None;
            }

            let task = self.pending.pop_front()?;
            let spill = &self.spills[task.spill_index];
            match prepare_partition_join(
                spill,
                task.partition,
                PreparePartitionContext {
                    left_keys: &self.left_keys,
                    right_keys: &self.right_keys,
                    memory_limit_bytes: self.memory_limit_bytes,
                    partitions: self.partitions,
                    depth: task.depth,
                    join_type: self.join_type,
                    metrics: &self.metrics,
                },
            ) {
                Ok(PartitionPreparation::Ready(Some(active))) => self.active = Some(*active),
                Ok(PartitionPreparation::Ready(None)) => continue,
                Ok(PartitionPreparation::Repartition) => {
                    let spill = match repartition_partition(
                        spill,
                        task.partition,
                        &self.left_keys,
                        &self.right_keys,
                        self.partitions,
                        task.depth + 1,
                        &self.metrics,
                    ) {
                        Ok(spill) => spill,
                        Err(error) => return Some(Err(error)),
                    };
                    let spill_index = self.spills.len();
                    self.spills.push(spill);
                    self.pending
                        .extend((0..self.partitions).map(|partition| PartitionTask {
                            spill_index,
                            partition,
                            depth: task.depth + 1,
                        }));
                    continue;
                }
                Err(error) => return Some(Err(error)),
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct PartitionTask {
    spill_index: usize,
    partition: usize,
    depth: usize,
}

enum PartitionPreparation {
    Ready(Option<Box<PartitionJoinState>>),
    Repartition,
}

enum PartitionJoinState {
    Hash(Box<HashPartitionJoinState>),
    FileHash(Box<FileHashPartitionJoinState>),
    BlockNestedLoop(Box<BlockNestedLoopJoinState>),
    UnmatchedSide(Box<UnmatchedSideJoinState>),
    Composite(VecDeque<PartitionJoinState>),
}

impl PartitionJoinState {
    fn next_output_batches(
        &mut self,
        left_prefix: &str,
        right_prefix: &str,
        output_projection: &JoinOutputProjection,
        metrics: &ScanPlanMetricsCounter,
    ) -> Result<Option<Vec<RecordBatch>>> {
        match self {
            Self::Hash(state) => {
                state.next_output_batches(left_prefix, right_prefix, output_projection, metrics)
            }
            Self::FileHash(state) => {
                state.next_output_batches(left_prefix, right_prefix, output_projection, metrics)
            }
            Self::BlockNestedLoop(state) => {
                state.next_output_batches(left_prefix, right_prefix, output_projection, metrics)
            }
            Self::UnmatchedSide(state) => {
                state.next_output_batches(left_prefix, right_prefix, metrics)
            }
            Self::Composite(states) => {
                while let Some(state) = states.front_mut() {
                    if let Some(batches) = state.next_output_batches(
                        left_prefix,
                        right_prefix,
                        output_projection,
                        metrics,
                    )? {
                        return Ok(Some(batches));
                    }
                    states.pop_front();
                }
                Ok(None)
            }
        }
    }
}

struct FileHashPartitionJoinState {
    build_stream: IpcBatchStream,
    pending_build_batches: VecDeque<RecordBatch>,
    memory_limit_bytes: u64,
    probe_files: Vec<PathBuf>,
    probe_stream: IpcBatchStream,
    current_build: Option<HashJoinBuild>,
    current_matched_build: MatchedBuildTracker,
    current_probe_template: Option<RecordBatch>,
    current_matched_probe_keys: HashSet<OwnedRow>,
    build_keys: Vec<String>,
    probe_keys: Vec<String>,
    build_side: JoinBuildSide,
    join_type: JoinType,
    build_schema: Option<Arc<Schema>>,
    matched_probe_keys: HashSet<OwnedRow>,
    emitted_unmatched_probe: bool,
    spill_dir: Option<PathBuf>,
}

impl Drop for FileHashPartitionJoinState {
    fn drop(&mut self) {
        if let Some(spill_dir) = &self.spill_dir {
            let _ = std::fs::remove_dir_all(spill_dir);
        }
    }
}

impl FileHashPartitionJoinState {
    fn next_output_batches(
        &mut self,
        left_prefix: &str,
        right_prefix: &str,
        output_projection: &JoinOutputProjection,
        metrics: &ScanPlanMetricsCounter,
    ) -> Result<Option<Vec<RecordBatch>>> {
        loop {
            if self.current_build.is_none() {
                let build_batches = self.next_build_batch_chunk()?;
                if build_batches.is_empty() {
                    return self.next_unmatched_probe_batches(left_prefix, right_prefix, metrics);
                }
                self.current_build = Some(build_hash_join_input(
                    &build_batches,
                    &self.build_keys,
                    HashBuildMode::Full,
                    true,
                    metrics,
                )?);
                self.current_matched_build = MatchedBuildTracker::for_build(
                    self.current_build.as_ref().expect("current build"),
                );
                self.current_probe_template = None;
                self.current_matched_probe_keys.clear();
                self.probe_stream = IpcBatchStream::new(self.probe_files.clone());
            }

            while let Some(probe) = self.probe_stream.next_batch()? {
                self.current_probe_template
                    .get_or_insert_with(|| probe.clone());
                let batches = probe_hash_join_batches(
                    &probe,
                    self.current_build.as_ref().expect("current build"),
                    &self.probe_keys,
                    left_prefix,
                    right_prefix,
                    self.build_side,
                    if self.join_type == JoinType::Semi {
                        JoinType::Semi
                    } else {
                        JoinType::Inner
                    },
                    (self.join_type == JoinType::Full).then_some(&mut self.current_matched_build),
                    (self.join_type == JoinType::Semi).then_some(&self.matched_probe_keys),
                    matches!(self.join_type, JoinType::Full | JoinType::Semi)
                        .then_some(&mut self.current_matched_probe_keys),
                    output_projection,
                    metrics,
                )?;
                if !batches.is_empty() {
                    return Ok(Some(batches));
                }
            }

            let unmatched_build = emit_unmatched_build_if_needed(
                self.current_build.as_ref().expect("current build"),
                Some(&self.current_matched_build),
                self.build_side,
                self.join_type,
                left_prefix,
                right_prefix,
                self.current_probe_template.as_ref(),
                metrics,
                &mut false,
            )?;
            if !unmatched_build.is_empty() {
                self.current_build = None;
                return Ok(Some(unmatched_build));
            }
            self.matched_probe_keys
                .extend(self.current_matched_probe_keys.drain());
            self.current_build = None;

            if self.pending_build_batches.is_empty() && self.build_stream.is_exhausted() {
                return self.next_unmatched_probe_batches(left_prefix, right_prefix, metrics);
            }
        }
    }

    fn next_build_batch_chunk(&mut self) -> Result<Vec<RecordBatch>> {
        let mut batches = Vec::new();
        let mut bytes = 0_u64;
        let limit = self.memory_limit_bytes.max(1);

        loop {
            let batch = if let Some(batch) = self.pending_build_batches.pop_front() {
                Some(batch)
            } else {
                self.build_stream.next_batch()?
            };
            let Some(batch) = batch else {
                break;
            };
            self.build_schema.get_or_insert_with(|| batch.schema());
            let batch_bytes = record_batch_memory_size(&batch).max(1);
            if !batches.is_empty() && bytes.saturating_add(batch_bytes) > limit {
                self.pending_build_batches.push_front(batch);
                break;
            }
            bytes = bytes.saturating_add(batch_bytes);
            batches.push(batch);
            if bytes >= limit {
                break;
            }
        }

        Ok(batches)
    }

    fn next_unmatched_probe_batches(
        &mut self,
        left_prefix: &str,
        right_prefix: &str,
        metrics: &ScanPlanMetricsCounter,
    ) -> Result<Option<Vec<RecordBatch>>> {
        if self.join_type != JoinType::Full || self.emitted_unmatched_probe {
            return Ok(None);
        }
        self.emitted_unmatched_probe = true;
        let build_schema = self.build_schema.as_ref().ok_or_else(|| {
            DodamError::UnsupportedSql(
                "FULL OUTER JOIN needs the build schema to emit unmatched probe rows".to_string(),
            )
        })?;
        let mut output = Vec::new();
        let mut probe_stream = IpcBatchStream::new(self.probe_files.clone());
        while let Some(probe) = probe_stream.next_batch()? {
            output.extend(materialize_unmatched_probe_by_matched_keys(
                &probe,
                &self.probe_keys,
                &self.matched_probe_keys,
                &UnmatchedProbeMaterializeContext {
                    build_schema,
                    build_side: self.build_side,
                    left_prefix,
                    right_prefix,
                    metrics,
                },
            )?);
        }
        if output.is_empty() {
            Ok(None)
        } else {
            Ok(Some(output))
        }
    }
}

struct HashPartitionJoinState {
    build: HashJoinBuild,
    probe_batches: Vec<RecordBatch>,
    next_probe: usize,
    probe_keys: Vec<String>,
    build_side: JoinBuildSide,
    join_type: JoinType,
    matched_build: MatchedBuildTracker,
    emitted_unmatched_build: bool,
    probe_template: Option<RecordBatch>,
}

impl HashPartitionJoinState {
    fn next_output_batches(
        &mut self,
        left_prefix: &str,
        right_prefix: &str,
        output_projection: &JoinOutputProjection,
        metrics: &ScanPlanMetricsCounter,
    ) -> Result<Option<Vec<RecordBatch>>> {
        while let Some(probe) = self.probe_batches.get(self.next_probe).cloned() {
            self.next_probe += 1;
            self.probe_template.get_or_insert_with(|| probe.clone());
            let batches = probe_hash_join_batches(
                &probe,
                &self.build,
                &self.probe_keys,
                left_prefix,
                right_prefix,
                self.build_side,
                self.join_type,
                (self.join_type == JoinType::Full).then_some(&mut self.matched_build),
                None,
                None,
                output_projection,
                metrics,
            )?;
            if !batches.is_empty() {
                return Ok(Some(batches));
            }
        }

        let batches = emit_unmatched_build_if_needed(
            &self.build,
            Some(&self.matched_build),
            self.build_side,
            self.join_type,
            left_prefix,
            right_prefix,
            self.probe_template.as_ref(),
            metrics,
            &mut self.emitted_unmatched_build,
        )?;
        if batches.is_empty() {
            Ok(None)
        } else {
            Ok(Some(batches))
        }
    }
}

struct UnmatchedSideJoinState {
    stream: IpcBatchStream,
    side: JoinBuildSide,
    opposite_schema: Arc<Schema>,
}

impl UnmatchedSideJoinState {
    fn next_output_batches(
        &mut self,
        left_prefix: &str,
        right_prefix: &str,
        metrics: &ScanPlanMetricsCounter,
    ) -> Result<Option<Vec<RecordBatch>>> {
        let Some(batch) = self.stream.next_batch()? else {
            return Ok(None);
        };
        let null_opposite = null_record_batch_for_schema(&self.opposite_schema, batch.num_rows())?;
        let (left, right) = match self.side {
            JoinBuildSide::Left => (batch, null_opposite),
            JoinBuildSide::Right => (null_opposite, batch),
        };
        metrics.add_join_output_rows(left.num_rows());
        Ok(Some(vec![join_output_batch(
            &left,
            &right,
            left_prefix,
            right_prefix,
        )?]))
    }
}

struct BlockNestedLoopJoinState {
    left_stream: IpcBatchStream,
    right_files: Vec<PathBuf>,
    right_stream: Option<IpcBatchStream>,
    current_left: Option<RecordBatch>,
    left_keys: Vec<String>,
    right_keys: Vec<String>,
    spill_dir: Option<PathBuf>,
}

impl Drop for BlockNestedLoopJoinState {
    fn drop(&mut self) {
        if let Some(spill_dir) = &self.spill_dir {
            let _ = std::fs::remove_dir_all(spill_dir);
        }
    }
}

impl BlockNestedLoopJoinState {
    fn next_output_batches(
        &mut self,
        left_prefix: &str,
        right_prefix: &str,
        output_projection: &JoinOutputProjection,
        metrics: &ScanPlanMetricsCounter,
    ) -> Result<Option<Vec<RecordBatch>>> {
        loop {
            if self.current_left.is_none() {
                let Some(left) = self.left_stream.next_batch()? else {
                    return Ok(None);
                };
                self.current_left = Some(left);
                self.right_stream = Some(IpcBatchStream::new(self.right_files.clone()));
            }

            let right_stream = self.right_stream.as_mut().expect("right stream");
            while let Some(right) = right_stream.next_batch()? {
                let batches = block_nested_loop_join_batches(
                    self.current_left.as_ref().expect("current left"),
                    &right,
                    &self.left_keys,
                    &self.right_keys,
                    left_prefix,
                    right_prefix,
                    output_projection,
                    metrics,
                )?;
                if !batches.is_empty() {
                    return Ok(Some(batches));
                }
            }

            self.current_left = None;
            self.right_stream = None;
        }
    }
}

struct IpcBatchStream {
    files: Vec<PathBuf>,
    next_file: usize,
    current: Option<IpcFileReader<File>>,
}

impl IpcBatchStream {
    fn new(files: Vec<PathBuf>) -> Self {
        Self {
            files,
            next_file: 0,
            current: None,
        }
    }

    fn is_exhausted(&self) -> bool {
        self.current.is_none() && self.next_file >= self.files.len()
    }

    fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
        loop {
            if self.current.is_none() {
                let Some(path) = self.files.get(self.next_file) else {
                    return Ok(None);
                };
                self.next_file += 1;
                self.current = Some(IpcFileReader::try_new(File::open(path)?, None)?);
            }

            let reader = self.current.as_mut().expect("current reader");
            match reader.next() {
                Some(batch) => {
                    let batch = batch?;
                    if batch.num_rows() == 0 {
                        continue;
                    }
                    return Ok(Some(batch));
                }
                None => self.current = None,
            }
        }
    }
}

struct SpilledJoin {
    dir: PathBuf,
    left: Vec<Vec<PathBuf>>,
    right: Vec<Vec<PathBuf>>,
    left_schema: Option<Arc<Schema>>,
    right_schema: Option<Arc<Schema>>,
}

struct SpillPartitionContext<'a> {
    partitions: usize,
    dir: &'a std::path::Path,
    side: &'a str,
    salt: u64,
    metrics: &'a ScanPlanMetricsCounter,
}

struct PreparePartitionContext<'a> {
    left_keys: &'a [String],
    right_keys: &'a [String],
    memory_limit_bytes: u64,
    partitions: usize,
    depth: usize,
    join_type: JoinType,
    metrics: &'a ScanPlanMetricsCounter,
}

impl Drop for SpilledJoin {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn collect_non_empty_batches(
    input: &mut Box<dyn Iterator<Item = Result<RecordBatch>> + Send>,
) -> Result<Vec<RecordBatch>> {
    let mut batches = Vec::new();
    for batch in input.by_ref() {
        let batch = batch?;
        if batch.num_rows() > 0 {
            batches.push(batch);
        }
    }
    Ok(batches)
}

#[derive(Debug, Clone)]
struct JoinOutputProjection {
    left_columns: Option<Vec<String>>,
    right_columns: Option<Vec<String>>,
}

impl JoinOutputProjection {
    fn all() -> Self {
        Self {
            left_columns: None,
            right_columns: None,
        }
    }

    fn from_projection(
        projection: &Projection,
        left_prefix: &str,
        right_prefix: &str,
        join_type: JoinType,
    ) -> Self {
        let Projection::Columns(columns) = projection else {
            return Self::all();
        };
        let mut output = Self {
            left_columns: Some(Vec::new()),
            right_columns: Some(Vec::new()),
        };
        for column in columns {
            if let Some(column) = strip_qualified_column(column, left_prefix) {
                add_column_once(output.left_columns.as_mut().expect("left columns"), column);
            } else if join_type != JoinType::Semi
                && let Some(column) = strip_qualified_column(column, right_prefix)
            {
                add_column_once(
                    output.right_columns.as_mut().expect("right columns"),
                    column,
                );
            }
        }
        output
    }
}

fn strip_qualified_column(column: &str, prefix: &str) -> Option<String> {
    column
        .strip_prefix(prefix)?
        .strip_prefix('.')
        .map(ToString::to_string)
}

fn add_column_once(columns: &mut Vec<String>, column: String) {
    if !columns.iter().any(|existing| existing == &column) {
        columns.push(column);
    }
}

struct HashJoinBuild {
    batches: Vec<RecordBatch>,
    key_data_types: Vec<DataType>,
    all_rows: Vec<BuildRowRef>,
    rows: HashMap<OwnedRow, Vec<BuildRowRef>>,
    heavy_rows: HashMap<OwnedRow, Vec<BuildRowRef>>,
    i32_rows: Option<JoinKeyHashMap<i32, Vec<BuildRowRef>>>,
    i32_dense_rows: Option<DenseI32MultiBuildRows>,
    i32_unique_rows: Option<JoinKeyHashMap<i32, BuildRowRef>>,
    i32_dense_unique_rows: Option<DenseI32BuildRows>,
    i32_key_set: Option<JoinKeyHashSet<i32>>,
    i32_dense_key_set: Option<DenseI32KeySet>,
    i32_pair_rows: Option<JoinKeyHashMap<(i32, i32), Vec<BuildRowRef>>>,
    i32_pair_unique_rows: Option<JoinKeyHashMap<(i32, i32), BuildRowRef>>,
    i32_pair_dense_unique_rows: Option<DenseI32PairBuildRows>,
    string_rows: Option<JoinKeyHashMap<String, Vec<BuildRowRef>>>,
    string_unique_rows: Option<JoinKeyHashMap<String, BuildRowRef>>,
    i64_rows: Option<JoinKeyHashMap<i64, Vec<BuildRowRef>>>,
    i64_unique_rows: Option<JoinKeyHashMap<i64, BuildRowRef>>,
    i64_dense_unique_rows: Option<DenseI64BuildRows>,
    bloom: BloomFilter,
}

#[derive(Debug, Clone)]
enum DenseI32BuildRows {
    Complete {
        min: i32,
        rows: Vec<BuildRowRef>,
    },
    Sparse {
        min: i32,
        rows: Vec<Option<BuildRowRef>>,
    },
}

impl DenseI32BuildRows {
    fn get(&self, key: i32) -> Option<BuildRowRef> {
        match self {
            Self::Complete { min, rows } => {
                let index = key.checked_sub(*min)? as usize;
                rows.get(index).copied()
            }
            Self::Sparse { min, rows } => {
                let index = key.checked_sub(*min)? as usize;
                rows.get(index).copied().flatten()
            }
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Complete { rows, .. } => rows.len(),
            Self::Sparse { rows, .. } => rows.len(),
        }
    }

    fn min(&self) -> i32 {
        match self {
            Self::Complete { min, .. } | Self::Sparse { min, .. } => *min,
        }
    }

    fn unmatched_refs<'a>(&'a self, matched: &'a [bool]) -> impl Iterator<Item = BuildRowRef> + 'a {
        match self {
            Self::Complete { rows, .. } => DenseI32UnmatchedRefs::Complete {
                rows: rows.iter().copied().enumerate(),
                matched,
            },
            Self::Sparse { rows, .. } => DenseI32UnmatchedRefs::Sparse {
                rows: rows.iter().copied().enumerate(),
                matched,
            },
        }
    }
}

enum DenseI32UnmatchedRefs<'a> {
    Complete {
        rows: std::iter::Enumerate<std::iter::Copied<std::slice::Iter<'a, BuildRowRef>>>,
        matched: &'a [bool],
    },
    Sparse {
        rows: std::iter::Enumerate<std::iter::Copied<std::slice::Iter<'a, Option<BuildRowRef>>>>,
        matched: &'a [bool],
    },
}

impl Iterator for DenseI32UnmatchedRefs<'_> {
    type Item = BuildRowRef;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Complete { rows, matched } => rows.find_map(|(index, row)| {
                (!matched.get(index).copied().unwrap_or(false)).then_some(row)
            }),
            Self::Sparse { rows, matched } => rows.find_map(|(index, row)| {
                (!matched.get(index).copied().unwrap_or(false))
                    .then_some(row)
                    .flatten()
            }),
        }
    }
}

#[derive(Debug, Clone)]
enum DenseI32PairBuildRows {
    Complete {
        min_left: i32,
        min_right: i32,
        right_width: usize,
        rows: Vec<BuildRowRef>,
    },
    Sparse {
        min_left: i32,
        min_right: i32,
        right_width: usize,
        rows: Vec<Option<BuildRowRef>>,
    },
}

impl DenseI32PairBuildRows {
    fn get(&self, left: i32, right: i32) -> Option<BuildRowRef> {
        match self {
            Self::Complete {
                min_left,
                min_right,
                right_width,
                rows,
            } => {
                let index = dense_i32_pair_index(left, right, *min_left, *min_right, *right_width)?;
                rows.get(index).copied()
            }
            Self::Sparse {
                min_left,
                min_right,
                right_width,
                rows,
            } => {
                let index = dense_i32_pair_index(left, right, *min_left, *min_right, *right_width)?;
                rows.get(index).copied().flatten()
            }
        }
    }
}

#[derive(Debug, Clone)]
enum DenseI32MultiBuildRows {
    Complete {
        min: i32,
        rows: Vec<Vec<BuildRowRef>>,
    },
    Sparse {
        min: i32,
        rows: Vec<Option<Vec<BuildRowRef>>>,
    },
}

impl DenseI32MultiBuildRows {
    fn get(&self, key: i32) -> Option<&[BuildRowRef]> {
        match self {
            Self::Complete { min, rows } => {
                let index = key.checked_sub(*min)? as usize;
                rows.get(index).map(Vec::as_slice)
            }
            Self::Sparse { min, rows } => {
                let index = key.checked_sub(*min)? as usize;
                rows.get(index).and_then(Option::as_deref)
            }
        }
    }
}

#[derive(Debug, Clone)]
struct DenseI32KeySet {
    min: i32,
    exists: Vec<bool>,
}

impl DenseI32KeySet {
    fn contains(&self, key: i32) -> bool {
        let Some(index) = key.checked_sub(self.min).map(|index| index as usize) else {
            return false;
        };
        self.exists.get(index).copied().unwrap_or(false)
    }
}

enum MatchedBuildTracker {
    Set(HashSet<BuildRowRef>),
    DenseI32 {
        min: i32,
        matched: Vec<bool>,
        matched_count: usize,
    },
}

impl MatchedBuildTracker {
    fn for_build(build: &HashJoinBuild) -> Self {
        if let Some(rows) = &build.i32_dense_unique_rows {
            Self::DenseI32 {
                min: rows.min(),
                matched: vec![false; rows.len()],
                matched_count: 0,
            }
        } else {
            Self::Set(HashSet::new())
        }
    }

    fn empty_set() -> Self {
        Self::Set(HashSet::new())
    }

    fn mark_ref(&mut self, build_ref: BuildRowRef) {
        if let Self::Set(rows) = self {
            rows.insert(build_ref);
        }
    }

    fn mark_refs(&mut self, matches: &[BuildRowRef]) {
        if let Self::Set(rows) = self {
            rows.extend(matches.iter().copied());
        }
    }

    fn mark_i32_key(&mut self, key: i32) {
        let Self::DenseI32 {
            min,
            matched,
            matched_count,
        } = self
        else {
            return;
        };
        let Some(index) = key.checked_sub(*min).map(|index| index as usize) else {
            return;
        };
        if let Some(is_matched) = matched.get_mut(index)
            && !*is_matched
        {
            *is_matched = true;
            *matched_count += 1;
        }
    }

    fn is_matched(&self, build_ref: &BuildRowRef) -> bool {
        match self {
            Self::Set(rows) => rows.contains(build_ref),
            Self::DenseI32 { .. } => false,
        }
    }

    fn all_dense_i32_matched(&self) -> bool {
        match self {
            Self::DenseI32 {
                matched,
                matched_count,
                ..
            } => *matched_count == matched.len(),
            Self::Set(_) => false,
        }
    }
}

#[derive(Debug, Clone)]
enum DenseI64BuildRows {
    Complete {
        min: i64,
        rows: Vec<BuildRowRef>,
    },
    Sparse {
        min: i64,
        rows: Vec<Option<BuildRowRef>>,
    },
}

impl DenseI64BuildRows {
    fn get(&self, key: i64) -> Option<BuildRowRef> {
        match self {
            Self::Complete { min, rows } => {
                let index = key.checked_sub(*min)? as usize;
                rows.get(index).copied()
            }
            Self::Sparse { min, rows } => {
                let index = key.checked_sub(*min)? as usize;
                rows.get(index).copied().flatten()
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HashBuildMode {
    Full,
    FastSingleKey,
    FastSemi,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct BuildRowRef {
    batch: usize,
    row: u32,
}

struct BloomFilter {
    bits: Vec<u64>,
    mask: u64,
}

impl BloomFilter {
    fn new(expected_rows: usize) -> Self {
        let bits = expected_rows.saturating_mul(16).next_power_of_two().max(64);
        Self {
            bits: vec![0; bits.div_ceil(64)],
            mask: (bits - 1) as u64,
        }
    }

    fn insert(&mut self, row: &OwnedRow) {
        for hash in bloom_hashes(row) {
            self.set(hash);
        }
    }

    fn might_contain(&self, row: &OwnedRow) -> bool {
        bloom_hashes(row).into_iter().all(|hash| self.get(hash))
    }

    fn set(&mut self, hash: u64) {
        let bit = hash & self.mask;
        let word = (bit / 64) as usize;
        let offset = bit % 64;
        self.bits[word] |= 1_u64 << offset;
    }

    fn get(&self, hash: u64) -> bool {
        let bit = hash & self.mask;
        let word = (bit / 64) as usize;
        let offset = bit % 64;
        (self.bits[word] & (1_u64 << offset)) != 0
    }
}

fn bloom_hashes(row: &OwnedRow) -> [u64; 3] {
    [
        hash_row_with_salt(row, 0x9e37_79b9_7f4a_7c15),
        hash_row_with_salt(row, 0xc2b2_ae3d_27d4_eb4f),
        hash_row_with_salt(row, 0x1656_67b1_9e37_79f9),
    ]
}

fn heavy_hitter_threshold(total_rows: usize) -> usize {
    (total_rows / 10).max(1024)
}

fn split_heavy_hitters(
    rows: HashMap<OwnedRow, Vec<BuildRowRef>>,
    total_rows: usize,
) -> (
    HashMap<OwnedRow, Vec<BuildRowRef>>,
    HashMap<OwnedRow, Vec<BuildRowRef>>,
) {
    let threshold = heavy_hitter_threshold(total_rows);
    let mut normal_rows = HashMap::new();
    let mut heavy_rows = HashMap::new();
    for (key, matches) in rows {
        if matches.len() >= threshold {
            heavy_rows.insert(key, matches);
        } else {
            normal_rows.insert(key, matches);
        }
    }
    (normal_rows, heavy_rows)
}

fn count_heavy_hitters(rows: &HashMap<OwnedRow, Vec<BuildRowRef>>) -> usize {
    rows.values().filter(|matches| !matches.is_empty()).count()
}

fn spill_join_inputs(
    left: &mut Box<dyn Iterator<Item = Result<RecordBatch>> + Send>,
    right: &mut Box<dyn Iterator<Item = Result<RecordBatch>> + Send>,
    left_keys: &[String],
    right_keys: &[String],
    partitions: usize,
    salt: u64,
    metrics: &ScanPlanMetricsCounter,
) -> Result<SpilledJoin> {
    let dir = create_spill_dir()?;
    let mut spill = SpilledJoin {
        dir,
        left: vec![Vec::new(); partitions],
        right: vec![Vec::new(); partitions],
        left_schema: None,
        right_schema: None,
    };

    spill.left_schema = spill_input_partitions(
        left,
        left_keys,
        &mut spill.left,
        SpillPartitionContext {
            partitions,
            dir: &spill.dir,
            side: "left",
            salt,
            metrics,
        },
    )?;
    spill.right_schema = spill_input_partitions(
        right,
        right_keys,
        &mut spill.right,
        SpillPartitionContext {
            partitions,
            dir: &spill.dir,
            side: "right",
            salt,
            metrics,
        },
    )?;
    Ok(spill)
}

fn spill_input_partitions(
    input: &mut Box<dyn Iterator<Item = Result<RecordBatch>> + Send>,
    keys: &[String],
    output: &mut [Vec<PathBuf>],
    context: SpillPartitionContext<'_>,
) -> Result<Option<Arc<Schema>>> {
    let mut sequence = 0_u64;
    let mut schema = None;
    for batch in input.by_ref() {
        let batch = batch?;
        if batch.num_rows() == 0 {
            continue;
        }
        schema.get_or_insert_with(|| batch.schema());
        let partitioned = partition_batch(&batch, keys, context.partitions, context.salt)?;
        for (partition, batch) in partitioned.into_iter().enumerate() {
            let Some(batch) = batch else {
                continue;
            };
            let path = context
                .dir
                .join(format!("{}-{partition}-{sequence}.arrow", context.side));
            sequence = sequence.saturating_add(1);
            write_ipc_batch(&path, &batch)?;
            context
                .metrics
                .add_join_spill_file(std::fs::metadata(&path)?.len());
            output[partition].push(path);
        }
    }
    Ok(schema)
}

fn partition_batch(
    batch: &RecordBatch,
    keys: &[String],
    partitions: usize,
    salt: u64,
) -> Result<Vec<Option<RecordBatch>>> {
    let key_arrays = key_arrays(batch, keys)?;
    let converter = RowConverter::new(
        key_arrays
            .iter()
            .map(|array| SortField::new(array.data_type().clone()))
            .collect(),
    )?;
    let rows = converter.convert_columns(&key_arrays)?;
    let mut indices = vec![Vec::<u32>::new(); partitions];

    for (row_index, row) in rows.iter().enumerate() {
        if key_arrays.iter().any(|array| array.is_null(row_index)) {
            continue;
        }
        let partition = hash_partition(&row.owned(), partitions, salt);
        let row_index = u32::try_from(row_index).map_err(|_| {
            DodamError::UnsupportedSql(
                "partitioned hash join currently supports up to u32::MAX rows per batch"
                    .to_string(),
            )
        })?;
        indices[partition].push(row_index);
    }

    indices
        .into_iter()
        .map(|indices| {
            if indices.is_empty() {
                return Ok(None);
            }
            Ok(Some(take_record_batch(batch, &UInt32Array::from(indices))?))
        })
        .collect()
}

fn hash_partition(row: &OwnedRow, partitions: usize, salt: u64) -> usize {
    (hash_row_with_salt(row, salt) as usize) % partitions
}

fn hash_row_with_salt(row: &OwnedRow, salt: u64) -> u64 {
    let mut hasher = DefaultHasher::new();
    salt.hash(&mut hasher);
    row.hash(&mut hasher);
    hasher.finish()
}

const MAX_REPARTITION_DEPTH: usize = 4;

fn prepare_partition_join(
    spill: &SpilledJoin,
    partition: usize,
    context: PreparePartitionContext<'_>,
) -> Result<PartitionPreparation> {
    let left_files = &spill.left[partition];
    let right_files = &spill.right[partition];
    if left_files.is_empty() || right_files.is_empty() {
        if context.join_type == JoinType::Full {
            if !left_files.is_empty() {
                let right_schema = spill.right_schema.clone().ok_or_else(|| {
                    DodamError::UnsupportedSql(
                        "FULL OUTER JOIN needs the right schema to emit unmatched left rows"
                            .to_string(),
                    )
                })?;
                return Ok(PartitionPreparation::Ready(Some(Box::new(
                    PartitionJoinState::UnmatchedSide(Box::new(UnmatchedSideJoinState {
                        stream: IpcBatchStream::new(left_files.clone()),
                        side: JoinBuildSide::Left,
                        opposite_schema: right_schema,
                    })),
                ))));
            }
            if !right_files.is_empty() {
                let left_schema = spill.left_schema.clone().ok_or_else(|| {
                    DodamError::UnsupportedSql(
                        "FULL OUTER JOIN needs the left schema to emit unmatched right rows"
                            .to_string(),
                    )
                })?;
                return Ok(PartitionPreparation::Ready(Some(Box::new(
                    PartitionJoinState::UnmatchedSide(Box::new(UnmatchedSideJoinState {
                        stream: IpcBatchStream::new(right_files.clone()),
                        side: JoinBuildSide::Right,
                        opposite_schema: left_schema,
                    })),
                ))));
            }
        }
        return Ok(PartitionPreparation::Ready(None));
    }

    let left_bytes = files_len(left_files)?;
    let right_bytes = files_len(right_files)?;
    if left_bytes.min(right_bytes) > context.memory_limit_bytes
        && context.depth < MAX_REPARTITION_DEPTH
        && context.partitions > 1
    {
        return Ok(PartitionPreparation::Repartition);
    }
    let (build_files, probe_files, build_keys, probe_keys, build_side) =
        if context.join_type == JoinType::Semi {
            (
                right_files,
                left_files,
                context.right_keys,
                context.left_keys,
                JoinBuildSide::Right,
            )
        } else if left_bytes <= right_bytes {
            (
                left_files,
                right_files,
                context.left_keys,
                context.right_keys,
                JoinBuildSide::Left,
            )
        } else {
            (
                right_files,
                left_files,
                context.right_keys,
                context.left_keys,
                JoinBuildSide::Right,
            )
        };

    let oversized = left_bytes.min(right_bytes) > context.memory_limit_bytes;
    if oversized {
        let heavy_keys = heavy_keys_from_files(build_files, build_keys)?;
        if heavy_keys.is_empty() {
            if matches!(context.join_type, JoinType::Full | JoinType::Semi) {
                return Ok(PartitionPreparation::Ready(Some(Box::new(
                    PartitionJoinState::FileHash(Box::new(FileHashPartitionJoinState {
                        build_stream: IpcBatchStream::new(build_files.clone()),
                        pending_build_batches: VecDeque::new(),
                        memory_limit_bytes: context.memory_limit_bytes,
                        probe_stream: IpcBatchStream::new(probe_files.clone()),
                        probe_files: probe_files.clone(),
                        current_build: None,
                        current_matched_build: MatchedBuildTracker::empty_set(),
                        current_probe_template: None,
                        current_matched_probe_keys: HashSet::new(),
                        build_keys: build_keys.to_vec(),
                        probe_keys: probe_keys.to_vec(),
                        build_side,
                        join_type: context.join_type,
                        build_schema: None,
                        matched_probe_keys: HashSet::new(),
                        emitted_unmatched_probe: false,
                        spill_dir: None,
                    })),
                ))));
            }
            context.metrics.add_join_nested_loop_fallback();
            return Ok(PartitionPreparation::Ready(Some(Box::new(
                PartitionJoinState::BlockNestedLoop(Box::new(BlockNestedLoopJoinState {
                    left_stream: IpcBatchStream::new(left_files.clone()),
                    right_files: right_files.clone(),
                    right_stream: None,
                    current_left: None,
                    left_keys: context.left_keys.to_vec(),
                    right_keys: context.right_keys.to_vec(),
                    spill_dir: None,
                })),
            ))));
        }
        context.metrics.add_join_heavy_hitters(heavy_keys.len());
        return Ok(PartitionPreparation::Ready(Some(Box::new(
            build_partition_join_states_from_files(
                build_files,
                probe_files,
                build_keys,
                probe_keys,
                build_side,
                context,
                &heavy_keys,
            )?,
        ))));
    }

    let build_batches = read_ipc_batches(build_files)?;
    if build_batches.is_empty() {
        return Ok(PartitionPreparation::Ready(None));
    }
    let probe_batches = read_ipc_batches(probe_files)?;
    if probe_batches.is_empty() {
        return Ok(PartitionPreparation::Ready(None));
    }

    Ok(PartitionPreparation::Ready(Some(Box::new(
        build_partition_join_states(
            build_batches,
            probe_batches,
            build_keys,
            probe_keys,
            build_side,
            context,
        )?,
    ))))
}

fn repartition_partition(
    spill: &SpilledJoin,
    partition: usize,
    left_keys: &[String],
    right_keys: &[String],
    partitions: usize,
    depth: usize,
    metrics: &ScanPlanMetricsCounter,
) -> Result<SpilledJoin> {
    metrics.add_join_repartition();
    let dir = create_spill_dir()?;
    let mut repartitioned = SpilledJoin {
        dir,
        left: vec![Vec::new(); partitions],
        right: vec![Vec::new(); partitions],
        left_schema: None,
        right_schema: None,
    };
    let salt = depth as u64;
    repartitioned.left_schema = repartition_files(
        &spill.left[partition],
        left_keys,
        &mut repartitioned.left,
        SpillPartitionContext {
            partitions,
            dir: &repartitioned.dir,
            side: "left",
            salt,
            metrics,
        },
    )?;
    repartitioned.right_schema = repartition_files(
        &spill.right[partition],
        right_keys,
        &mut repartitioned.right,
        SpillPartitionContext {
            partitions,
            dir: &repartitioned.dir,
            side: "right",
            salt,
            metrics,
        },
    )?;
    Ok(repartitioned)
}

fn build_partition_join_states(
    build_batches: Vec<RecordBatch>,
    probe_batches: Vec<RecordBatch>,
    build_keys: &[String],
    probe_keys: &[String],
    build_side: JoinBuildSide,
    context: PreparePartitionContext<'_>,
) -> Result<PartitionJoinState> {
    let heavy_keys = heavy_keys_from_batches(&build_batches, build_keys)?;
    if heavy_keys.is_empty() {
        return Ok(PartitionJoinState::Hash(Box::new(HashPartitionJoinState {
            build: build_hash_join_input(
                &build_batches,
                build_keys,
                HashBuildMode::Full,
                true,
                context.metrics,
            )?,
            probe_batches,
            next_probe: 0,
            probe_keys: probe_keys.to_vec(),
            build_side,
            join_type: context.join_type,
            matched_build: MatchedBuildTracker::empty_set(),
            emitted_unmatched_build: false,
            probe_template: None,
        })));
    }
    context.metrics.add_join_heavy_hitters(heavy_keys.len());

    let (normal_build_batches, heavy_build_batches) =
        split_batches_by_key_set(build_batches, build_keys, &heavy_keys)?;
    let (normal_probe_batches, heavy_probe_batches) =
        split_batches_by_key_set(probe_batches, probe_keys, &heavy_keys)?;
    let mut states = VecDeque::new();

    if !normal_build_batches.is_empty() && !normal_probe_batches.is_empty() {
        states.push_back(PartitionJoinState::Hash(Box::new(HashPartitionJoinState {
            build: build_hash_join_input(
                &normal_build_batches,
                build_keys,
                HashBuildMode::Full,
                true,
                context.metrics,
            )?,
            probe_batches: normal_probe_batches,
            next_probe: 0,
            probe_keys: probe_keys.to_vec(),
            build_side,
            join_type: context.join_type,
            matched_build: MatchedBuildTracker::empty_set(),
            emitted_unmatched_build: false,
            probe_template: None,
        })));
    }

    if !heavy_build_batches.is_empty() && !heavy_probe_batches.is_empty() {
        let spill_dir = create_spill_dir()?;
        let heavy_build_files = write_temp_batches(
            &spill_dir,
            "heavy-build",
            &heavy_build_batches,
            context.metrics,
        )?;
        let heavy_probe_files = write_temp_batches(
            &spill_dir,
            "heavy-probe",
            &heavy_probe_batches,
            context.metrics,
        )?;
        let (left_files, right_files, left_keys, right_keys) = match build_side {
            JoinBuildSide::Left => (
                heavy_build_files,
                heavy_probe_files,
                build_keys.to_vec(),
                probe_keys.to_vec(),
            ),
            JoinBuildSide::Right => (
                heavy_probe_files,
                heavy_build_files,
                probe_keys.to_vec(),
                build_keys.to_vec(),
            ),
        };
        if matches!(context.join_type, JoinType::Full | JoinType::Semi) {
            let (build_files, probe_files, build_keys, probe_keys) = match build_side {
                JoinBuildSide::Left => (
                    left_files,
                    right_files,
                    left_keys.clone(),
                    right_keys.clone(),
                ),
                JoinBuildSide::Right => (
                    right_files,
                    left_files,
                    right_keys.clone(),
                    left_keys.clone(),
                ),
            };
            states.push_back(PartitionJoinState::FileHash(Box::new(
                FileHashPartitionJoinState {
                    build_stream: IpcBatchStream::new(build_files),
                    pending_build_batches: VecDeque::new(),
                    memory_limit_bytes: context.memory_limit_bytes,
                    probe_stream: IpcBatchStream::new(probe_files.clone()),
                    probe_files,
                    current_build: None,
                    current_matched_build: MatchedBuildTracker::empty_set(),
                    current_probe_template: None,
                    current_matched_probe_keys: HashSet::new(),
                    build_keys,
                    probe_keys,
                    build_side,
                    join_type: context.join_type,
                    build_schema: None,
                    matched_probe_keys: HashSet::new(),
                    emitted_unmatched_probe: false,
                    spill_dir: Some(spill_dir),
                },
            )));
        } else {
            states.push_back(PartitionJoinState::BlockNestedLoop(Box::new(
                BlockNestedLoopJoinState {
                    left_stream: IpcBatchStream::new(left_files),
                    right_files,
                    right_stream: None,
                    current_left: None,
                    left_keys,
                    right_keys,
                    spill_dir: Some(spill_dir),
                },
            )));
        }
    }

    partition_join_state_from_parts(states)
}

fn build_partition_join_states_from_files(
    build_files: &[PathBuf],
    probe_files: &[PathBuf],
    build_keys: &[String],
    probe_keys: &[String],
    build_side: JoinBuildSide,
    context: PreparePartitionContext<'_>,
    heavy_keys: &HashSet<OwnedRow>,
) -> Result<PartitionJoinState> {
    let spill_dir = create_spill_dir()?;
    let (normal_build_files, heavy_build_files) = split_files_by_key_set(
        build_files,
        build_keys,
        heavy_keys,
        &spill_dir,
        "split-build",
        context.metrics,
    )?;
    let (normal_probe_files, heavy_probe_files) = split_files_by_key_set(
        probe_files,
        probe_keys,
        heavy_keys,
        &spill_dir,
        "split-probe",
        context.metrics,
    )?;
    let mut states = VecDeque::new();
    let has_heavy_state = !heavy_build_files.is_empty() && !heavy_probe_files.is_empty();

    if !normal_build_files.is_empty() && !normal_probe_files.is_empty() {
        states.push_back(PartitionJoinState::FileHash(Box::new(
            FileHashPartitionJoinState {
                build_stream: IpcBatchStream::new(normal_build_files),
                pending_build_batches: VecDeque::new(),
                memory_limit_bytes: context.memory_limit_bytes,
                probe_stream: IpcBatchStream::new(normal_probe_files.clone()),
                probe_files: normal_probe_files,
                current_build: None,
                current_matched_build: MatchedBuildTracker::empty_set(),
                current_probe_template: None,
                current_matched_probe_keys: HashSet::new(),
                build_keys: build_keys.to_vec(),
                probe_keys: probe_keys.to_vec(),
                build_side,
                join_type: context.join_type,
                build_schema: None,
                matched_probe_keys: HashSet::new(),
                emitted_unmatched_probe: false,
                spill_dir: (!has_heavy_state).then_some(spill_dir.clone()),
            },
        )));
    }

    if has_heavy_state {
        let (left_files, right_files, left_keys, right_keys) = match build_side {
            JoinBuildSide::Left => (
                heavy_build_files,
                heavy_probe_files,
                build_keys.to_vec(),
                probe_keys.to_vec(),
            ),
            JoinBuildSide::Right => (
                heavy_probe_files,
                heavy_build_files,
                probe_keys.to_vec(),
                build_keys.to_vec(),
            ),
        };
        if matches!(context.join_type, JoinType::Full | JoinType::Semi) {
            let (build_files, probe_files, build_keys, probe_keys) = match build_side {
                JoinBuildSide::Left => (
                    left_files,
                    right_files,
                    left_keys.clone(),
                    right_keys.clone(),
                ),
                JoinBuildSide::Right => (
                    right_files,
                    left_files,
                    right_keys.clone(),
                    left_keys.clone(),
                ),
            };
            states.push_back(PartitionJoinState::FileHash(Box::new(
                FileHashPartitionJoinState {
                    build_stream: IpcBatchStream::new(build_files),
                    pending_build_batches: VecDeque::new(),
                    memory_limit_bytes: context.memory_limit_bytes,
                    probe_stream: IpcBatchStream::new(probe_files.clone()),
                    probe_files,
                    current_build: None,
                    current_matched_build: MatchedBuildTracker::empty_set(),
                    current_probe_template: None,
                    current_matched_probe_keys: HashSet::new(),
                    build_keys,
                    probe_keys,
                    build_side,
                    join_type: context.join_type,
                    build_schema: None,
                    matched_probe_keys: HashSet::new(),
                    emitted_unmatched_probe: false,
                    spill_dir: Some(spill_dir),
                },
            )));
        } else {
            states.push_back(PartitionJoinState::BlockNestedLoop(Box::new(
                BlockNestedLoopJoinState {
                    left_stream: IpcBatchStream::new(left_files),
                    right_files,
                    right_stream: None,
                    current_left: None,
                    left_keys,
                    right_keys,
                    spill_dir: Some(spill_dir),
                },
            )));
        }
    } else if states.is_empty() {
        let _ = std::fs::remove_dir_all(&spill_dir);
    }

    partition_join_state_from_parts(states)
}

fn partition_join_state_from_parts(
    mut states: VecDeque<PartitionJoinState>,
) -> Result<PartitionJoinState> {
    match states.len() {
        0 => Ok(PartitionJoinState::Composite(states)),
        1 => Ok(states.pop_front().expect("one state")),
        _ => Ok(PartitionJoinState::Composite(states)),
    }
}

fn heavy_keys_from_batches(batches: &[RecordBatch], keys: &[String]) -> Result<HashSet<OwnedRow>> {
    let build = build_key_counts(batches, keys)?;
    let threshold = heavy_hitter_threshold(build.total_rows);
    Ok(build
        .counts
        .into_iter()
        .filter_map(|(key, count)| (count >= threshold).then_some(key))
        .collect())
}

fn heavy_keys_from_files(paths: &[PathBuf], keys: &[String]) -> Result<HashSet<OwnedRow>> {
    let build = build_key_counts_from_files(paths, keys)?;
    let threshold = heavy_hitter_threshold(build.total_rows);
    Ok(build
        .counts
        .into_iter()
        .filter_map(|(key, count)| (count >= threshold).then_some(key))
        .collect())
}

struct KeyCounts {
    counts: HashMap<OwnedRow, usize>,
    total_rows: usize,
}

fn build_key_counts(batches: &[RecordBatch], keys: &[String]) -> Result<KeyCounts> {
    let mut counts = HashMap::<OwnedRow, usize>::new();
    let mut total_rows = 0_usize;
    for batch in batches {
        add_key_counts(batch, keys, &mut counts, &mut total_rows)?;
    }
    Ok(KeyCounts { counts, total_rows })
}

fn build_key_counts_from_files(paths: &[PathBuf], keys: &[String]) -> Result<KeyCounts> {
    let mut counts = HashMap::<OwnedRow, usize>::new();
    let mut total_rows = 0_usize;
    for path in paths {
        let file = File::open(path)?;
        let reader = IpcFileReader::try_new(file, None)?;
        for batch in reader {
            let batch = batch?;
            add_key_counts(&batch, keys, &mut counts, &mut total_rows)?;
        }
    }
    Ok(KeyCounts { counts, total_rows })
}

fn add_key_counts(
    batch: &RecordBatch,
    keys: &[String],
    counts: &mut HashMap<OwnedRow, usize>,
    total_rows: &mut usize,
) -> Result<()> {
    *total_rows = total_rows.saturating_add(batch.num_rows());
    let key_arrays = key_arrays(batch, keys)?;
    let converter = RowConverter::new(
        key_arrays
            .iter()
            .map(|array| SortField::new(array.data_type().clone()))
            .collect(),
    )?;
    let rows = converter.convert_columns(&key_arrays)?;
    for (row_index, row) in rows.iter().enumerate() {
        if key_arrays.iter().any(|array| array.is_null(row_index)) {
            continue;
        }
        *counts.entry(row.owned()).or_default() += 1;
    }
    Ok(())
}

fn split_batches_by_key_set(
    batches: Vec<RecordBatch>,
    keys: &[String],
    heavy_keys: &HashSet<OwnedRow>,
) -> Result<(Vec<RecordBatch>, Vec<RecordBatch>)> {
    let mut normal = Vec::new();
    let mut heavy = Vec::new();
    for batch in batches {
        let (normal_batch, heavy_batch) = split_batch_by_key_set(batch, keys, heavy_keys)?;
        if let Some(batch) = normal_batch {
            normal.push(batch);
        }
        if let Some(batch) = heavy_batch {
            heavy.push(batch);
        }
    }
    Ok((normal, heavy))
}

fn split_batch_by_key_set(
    batch: RecordBatch,
    keys: &[String],
    heavy_keys: &HashSet<OwnedRow>,
) -> Result<(Option<RecordBatch>, Option<RecordBatch>)> {
    let key_arrays = key_arrays(&batch, keys)?;
    let converter = RowConverter::new(
        key_arrays
            .iter()
            .map(|array| SortField::new(array.data_type().clone()))
            .collect(),
    )?;
    let rows = converter.convert_columns(&key_arrays)?;
    let mut normal_indices = Vec::new();
    let mut heavy_indices = Vec::new();
    for (row_index, row) in rows.iter().enumerate() {
        if key_arrays.iter().any(|array| array.is_null(row_index)) {
            continue;
        }
        let index = u32::try_from(row_index).map_err(|_| {
            DodamError::UnsupportedSql(
                "heavy key split currently supports up to u32::MAX rows per batch".to_string(),
            )
        })?;
        if heavy_keys.contains(&row.owned()) {
            heavy_indices.push(index);
        } else {
            normal_indices.push(index);
        }
    }
    let normal = if normal_indices.is_empty() {
        None
    } else {
        Some(take_record_batch(
            &batch,
            &UInt32Array::from(normal_indices),
        )?)
    };
    let heavy = if heavy_indices.is_empty() {
        None
    } else {
        Some(take_record_batch(
            &batch,
            &UInt32Array::from(heavy_indices),
        )?)
    };
    Ok((normal, heavy))
}

fn split_files_by_key_set(
    paths: &[PathBuf],
    keys: &[String],
    heavy_keys: &HashSet<OwnedRow>,
    dir: &std::path::Path,
    prefix: &str,
    metrics: &ScanPlanMetricsCounter,
) -> Result<(Vec<PathBuf>, Vec<PathBuf>)> {
    let mut normal_files = Vec::new();
    let mut heavy_files = Vec::new();
    let mut sequence = 0_usize;
    for path in paths {
        let file = File::open(path)?;
        let reader = IpcFileReader::try_new(file, None)?;
        for batch in reader {
            let batch = batch?;
            let (normal_batch, heavy_batch) = split_batch_by_key_set(batch, keys, heavy_keys)?;
            if let Some(batch) = normal_batch {
                let path = dir.join(format!("{prefix}-normal-{sequence}.arrow"));
                sequence = sequence.saturating_add(1);
                write_ipc_batch(&path, &batch)?;
                metrics.add_join_spill_file(std::fs::metadata(&path)?.len());
                normal_files.push(path);
            }
            if let Some(batch) = heavy_batch {
                let path = dir.join(format!("{prefix}-heavy-{sequence}.arrow"));
                sequence = sequence.saturating_add(1);
                write_ipc_batch(&path, &batch)?;
                metrics.add_join_spill_file(std::fs::metadata(&path)?.len());
                heavy_files.push(path);
            }
        }
    }
    Ok((normal_files, heavy_files))
}

fn write_temp_batches(
    dir: &std::path::Path,
    prefix: &str,
    batches: &[RecordBatch],
    metrics: &ScanPlanMetricsCounter,
) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for (index, batch) in batches.iter().enumerate() {
        let path = dir.join(format!("{prefix}-{index}.arrow"));
        write_ipc_batch(&path, batch)?;
        metrics.add_join_spill_file(std::fs::metadata(&path)?.len());
        paths.push(path);
    }
    Ok(paths)
}

fn repartition_files(
    input_files: &[PathBuf],
    keys: &[String],
    output: &mut [Vec<PathBuf>],
    context: SpillPartitionContext<'_>,
) -> Result<Option<Arc<Schema>>> {
    let mut sequence = 0_u64;
    let mut schema = None;
    for path in input_files {
        let file = File::open(path)?;
        let reader = IpcFileReader::try_new(file, None)?;
        for batch in reader {
            let batch = batch?;
            if batch.num_rows() == 0 {
                continue;
            }
            schema.get_or_insert_with(|| batch.schema());
            let partitioned = partition_batch(&batch, keys, context.partitions, context.salt)?;
            for (partition, batch) in partitioned.into_iter().enumerate() {
                let Some(batch) = batch else {
                    continue;
                };
                let path = context
                    .dir
                    .join(format!("{}-{partition}-{sequence}.arrow", context.side));
                sequence = sequence.saturating_add(1);
                write_ipc_batch(&path, &batch)?;
                context
                    .metrics
                    .add_join_spill_file(std::fs::metadata(&path)?.len());
                output[partition].push(path);
            }
        }
    }
    Ok(schema)
}

fn files_len(paths: &[PathBuf]) -> Result<u64> {
    paths.iter().try_fold(0_u64, |total, path| {
        Ok(total.saturating_add(std::fs::metadata(path)?.len()))
    })
}

fn record_batch_memory_size(batch: &RecordBatch) -> u64 {
    batch.get_array_memory_size().min(u64::MAX as usize) as u64
}

fn key_arrays(batch: &RecordBatch, keys: &[String]) -> Result<Vec<ArrayRef>> {
    if keys.is_empty() {
        return Err(DodamError::UnsupportedSql(
            "JOIN requires at least one key column".to_string(),
        ));
    }
    keys.iter()
        .map(|key| Ok(batch.column(column_index(batch, key)?).clone()))
        .collect()
}

fn key_data_types(batch: &RecordBatch, keys: &[String]) -> Result<Vec<DataType>> {
    key_arrays(batch, keys).map(|arrays| {
        arrays
            .iter()
            .map(|array| array.data_type().clone())
            .collect()
    })
}

fn write_ipc_batch(path: &std::path::Path, batch: &RecordBatch) -> Result<()> {
    let mut file = File::create(path)?;
    let mut writer = IpcFileWriter::try_new(&mut file, batch.schema().as_ref())?;
    writer.write(batch)?;
    writer.finish()?;
    Ok(())
}

fn read_ipc_batches(paths: &[PathBuf]) -> Result<Vec<RecordBatch>> {
    let mut batches = Vec::new();
    for path in paths {
        let file = File::open(path)?;
        let reader = IpcFileReader::try_new(file, None)?;
        for batch in reader {
            let batch = batch?;
            if batch.num_rows() > 0 {
                batches.push(batch);
            }
        }
    }
    Ok(batches)
}

static SPILL_DIR_COUNTER: AtomicU64 = AtomicU64::new(0);

fn create_spill_dir() -> Result<PathBuf> {
    let sequence = SPILL_DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "dodam-join-spill-{}-{sequence}",
        std::process::id()
    ));
    std::fs::create_dir(&dir)?;
    Ok(dir)
}

fn build_hash_join_input(
    batches: &[RecordBatch],
    keys: &[String],
    mode: HashBuildMode,
    collect_all_rows: bool,
    metrics: &ScanPlanMetricsCounter,
) -> Result<HashJoinBuild> {
    let key_data_types = key_data_types(&batches[0], keys)?;
    let can_build_i32_only =
        mode == HashBuildMode::FastSingleKey && key_data_types.as_slice() == [DataType::Int32];
    let can_build_i64_only =
        mode == HashBuildMode::FastSingleKey && key_data_types.as_slice() == [DataType::Int64];
    let can_build_i32_key_set =
        mode == HashBuildMode::FastSemi && key_data_types.as_slice() == [DataType::Int32];
    let can_build_i32_pair = key_data_types.as_slice() == [DataType::Int32, DataType::Int32];
    let can_build_string = key_data_types.as_slice() == [DataType::Utf8];
    let total_rows = batches.iter().map(RecordBatch::num_rows).sum::<usize>();
    let build_bytes = batches
        .iter()
        .map(record_batch_memory_size)
        .fold(0_u64, u64::saturating_add);
    if can_build_i32_only
        && let Some(build) = try_build_dense_i32_hash_join_input(
            batches,
            keys,
            collect_all_rows,
            metrics,
            build_bytes,
        )?
    {
        return Ok(build);
    }
    if mode == HashBuildMode::FastSingleKey
        && can_build_i32_pair
        && let Some(build) = try_build_dense_i32_pair_hash_join_input(
            batches,
            keys,
            collect_all_rows,
            metrics,
            build_bytes,
        )?
    {
        return Ok(build);
    }
    let mut all_rows = collect_all_rows
        .then(|| Vec::with_capacity(total_rows))
        .unwrap_or_default();
    let mut rows = HashMap::<OwnedRow, Vec<BuildRowRef>>::new();
    let mut i32_rows: Option<JoinKeyHashMap<i32, Vec<BuildRowRef>>> = (key_data_types.as_slice()
        == [DataType::Int32]
        && !can_build_i32_key_set
        && !can_build_i32_only)
        .then(JoinKeyHashMap::default);
    let mut i32_unique_rows: Option<JoinKeyHashMap<i32, BuildRowRef>> =
        can_build_i32_only.then(JoinKeyHashMap::default);
    let mut i32_key_set: Option<JoinKeyHashSet<i32>> =
        can_build_i32_key_set.then(JoinKeyHashSet::default);
    let mut i32_pair_rows: Option<JoinKeyHashMap<(i32, i32), Vec<BuildRowRef>>> = None;
    let mut i32_pair_unique_rows: Option<JoinKeyHashMap<(i32, i32), BuildRowRef>> =
        can_build_i32_pair.then(JoinKeyHashMap::default);
    let mut string_rows: Option<JoinKeyHashMap<String, Vec<BuildRowRef>>> = None;
    let mut string_unique_rows: Option<JoinKeyHashMap<String, BuildRowRef>> =
        can_build_string.then(JoinKeyHashMap::default);
    let mut i64_rows: Option<JoinKeyHashMap<i64, Vec<BuildRowRef>>> =
        (key_data_types.as_slice() == [DataType::Int64] && !can_build_i64_only)
            .then(JoinKeyHashMap::default);
    let mut i64_unique_rows: Option<JoinKeyHashMap<i64, BuildRowRef>> =
        can_build_i64_only.then(JoinKeyHashMap::default);
    let mut bloom = BloomFilter::new(total_rows.max(1));

    for (batch_index, batch) in batches.iter().enumerate() {
        let key_arrays = key_arrays(batch, keys)?;
        let i32_key_array = if i32_rows.is_some() || i32_unique_rows.is_some() {
            Some(
                key_arrays[0]
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .expect("Int32 key array"),
            )
        } else {
            None
        };
        let i64_key_array = if i64_rows.is_some() || i64_unique_rows.is_some() {
            Some(
                key_arrays[0]
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .expect("Int64 key array"),
            )
        } else {
            None
        };
        let i32_pair_key_arrays = if i32_pair_rows.is_some() || i32_pair_unique_rows.is_some() {
            Some((
                key_arrays[0]
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .expect("first Int32 key array"),
                key_arrays[1]
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .expect("second Int32 key array"),
            ))
        } else {
            None
        };
        let string_key_array = if string_rows.is_some() || string_unique_rows.is_some() {
            Some(
                key_arrays[0]
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .expect("Utf8 key array"),
            )
        } else {
            None
        };
        for (key, data_type) in keys.iter().zip(&key_data_types) {
            let key_array = batch.column(column_index(batch, key)?);
            if key_array.data_type() != data_type {
                return Err(DodamError::UnsupportedSql(format!(
                    "JOIN key types must match across build batches: {} is {:?}, expected {:?}",
                    key,
                    key_array.data_type(),
                    data_type
                )));
            }
        }

        if can_build_i32_only {
            let i32_key_array = i32_key_array.expect("Int32 key array");
            for row in 0..batch.num_rows() {
                let row_index = u32::try_from(row).map_err(|_| {
                    DodamError::UnsupportedSql(
                        "hash join currently supports up to u32::MAX rows per batch".to_string(),
                    )
                })?;
                let row_ref = BuildRowRef {
                    batch: batch_index,
                    row: row_index,
                };
                if collect_all_rows {
                    all_rows.push(row_ref);
                }
                if i32_key_array.is_null(row) {
                    continue;
                }
                insert_i32_build_ref(
                    i32_key_array.value(row),
                    row_ref,
                    &mut i32_unique_rows,
                    &mut i32_rows,
                );
            }
            continue;
        }

        if can_build_i64_only {
            let i64_key_array = i64_key_array.expect("Int64 key array");
            for row in 0..batch.num_rows() {
                let row_index = u32::try_from(row).map_err(|_| {
                    DodamError::UnsupportedSql(
                        "hash join currently supports up to u32::MAX rows per batch".to_string(),
                    )
                })?;
                let row_ref = BuildRowRef {
                    batch: batch_index,
                    row: row_index,
                };
                if collect_all_rows {
                    all_rows.push(row_ref);
                }
                if i64_key_array.is_null(row) {
                    continue;
                }
                insert_i64_build_ref(
                    i64_key_array.value(row),
                    row_ref,
                    &mut i64_unique_rows,
                    &mut i64_rows,
                );
            }
            continue;
        }

        if can_build_i32_key_set {
            let i32_key_array = key_arrays[0]
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("Int32 key array");
            let i32_key_set = i32_key_set.as_mut().expect("Int32 key set");
            for row in 0..batch.num_rows() {
                if !i32_key_array.is_null(row) {
                    i32_key_set.insert(i32_key_array.value(row));
                }
            }
            continue;
        }

        if mode == HashBuildMode::FastSingleKey && (can_build_i32_pair || can_build_string) {
            for row in 0..batch.num_rows() {
                let row_index = u32::try_from(row).map_err(|_| {
                    DodamError::UnsupportedSql(
                        "hash join currently supports up to u32::MAX rows per batch".to_string(),
                    )
                })?;
                let row_ref = BuildRowRef {
                    batch: batch_index,
                    row: row_index,
                };
                if collect_all_rows {
                    all_rows.push(row_ref);
                }
                if key_arrays.iter().any(|array| array.is_null(row)) {
                    continue;
                }
                if let Some((left_key_array, right_key_array)) = i32_pair_key_arrays {
                    insert_i32_pair_build_ref(
                        (left_key_array.value(row), right_key_array.value(row)),
                        row_ref,
                        &mut i32_pair_unique_rows,
                        &mut i32_pair_rows,
                    );
                }
                if let Some(string_key_array) = string_key_array {
                    insert_string_build_ref(
                        string_key_array.value(row),
                        row_ref,
                        &mut string_unique_rows,
                        &mut string_rows,
                    );
                }
            }
            continue;
        }

        let converter = RowConverter::new(
            key_arrays
                .iter()
                .map(|array| SortField::new(array.data_type().clone()))
                .collect(),
        )?;
        let key_rows = converter.convert_columns(&key_arrays)?;

        for (row, key) in key_rows.iter().enumerate() {
            let row_index = u32::try_from(row).map_err(|_| {
                DodamError::UnsupportedSql(
                    "hash join currently supports up to u32::MAX rows per batch".to_string(),
                )
            })?;
            let row_ref = BuildRowRef {
                batch: batch_index,
                row: row_index,
            };
            if collect_all_rows {
                all_rows.push(row_ref);
            }
            if key_arrays.iter().any(|array| array.is_null(row)) {
                continue;
            }
            if let (Some(i32_rows), Some(i32_key_array)) = (&mut i32_rows, i32_key_array) {
                i32_rows
                    .entry(i32_key_array.value(row))
                    .or_default()
                    .push(row_ref);
            }
            if let (Some(i64_rows), Some(i64_key_array)) = (&mut i64_rows, i64_key_array) {
                i64_rows
                    .entry(i64_key_array.value(row))
                    .or_default()
                    .push(row_ref);
            }
            if let Some((left_key_array, right_key_array)) = i32_pair_key_arrays {
                insert_i32_pair_build_ref(
                    (left_key_array.value(row), right_key_array.value(row)),
                    row_ref,
                    &mut i32_pair_unique_rows,
                    &mut i32_pair_rows,
                );
            }
            if let Some(string_key_array) = string_key_array {
                insert_string_build_ref(
                    string_key_array.value(row),
                    row_ref,
                    &mut string_unique_rows,
                    &mut string_rows,
                );
            }
            let key = key.owned();
            bloom.insert(&key);
            rows.entry(key).or_default().push(row_ref);
        }
    }
    let i32_dense_unique_rows = i32_unique_rows
        .as_ref()
        .and_then(|rows| dense_i32_build_rows(rows, total_rows));
    let i32_dense_rows = i32_rows
        .as_ref()
        .and_then(|rows| dense_i32_multi_build_rows(rows, total_rows));
    let i32_dense_key_set = i32_key_set
        .as_ref()
        .and_then(|keys| dense_i32_key_set(keys, total_rows));
    let i32_pair_dense_unique_rows = i32_pair_unique_rows
        .as_ref()
        .and_then(|rows| dense_i32_pair_build_rows(rows, total_rows));
    let i64_dense_unique_rows = i64_unique_rows
        .as_ref()
        .and_then(|rows| dense_i64_build_rows(rows, total_rows));
    let (rows, heavy_rows) = split_heavy_hitters(rows, total_rows);
    metrics.add_join_build_rows(total_rows);
    metrics.observe_join_build_bytes(build_bytes);
    metrics.add_join_heavy_hitters(count_heavy_hitters(&heavy_rows));

    Ok(HashJoinBuild {
        key_data_types,
        batches: batches.to_vec(),
        all_rows,
        rows,
        heavy_rows,
        i32_rows,
        i32_dense_rows,
        i32_unique_rows,
        i32_dense_unique_rows,
        i32_key_set,
        i32_dense_key_set,
        i32_pair_rows,
        i32_pair_unique_rows,
        i32_pair_dense_unique_rows,
        string_rows,
        string_unique_rows,
        i64_rows,
        i64_unique_rows,
        i64_dense_unique_rows,
        bloom,
    })
}

fn try_build_dense_i32_hash_join_input(
    batches: &[RecordBatch],
    keys: &[String],
    collect_all_rows: bool,
    metrics: &ScanPlanMetricsCounter,
    build_bytes: u64,
) -> Result<Option<HashJoinBuild>> {
    let [key] = keys else {
        return Ok(None);
    };
    let mut min_key = i32::MAX;
    let mut max_key = i32::MIN;
    let mut non_null_rows = 0_usize;
    let total_rows = batches.iter().map(RecordBatch::num_rows).sum::<usize>();
    let mut all_rows = collect_all_rows
        .then(|| Vec::with_capacity(total_rows))
        .unwrap_or_default();

    for (batch_index, batch) in batches.iter().enumerate() {
        let key_index = column_index(batch, key)?;
        let key_array = batch.column(key_index);
        if key_array.data_type() != &DataType::Int32 {
            return Ok(None);
        }
        let key_array = key_array
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("Int32 key array");
        for row in 0..batch.num_rows() {
            let row_index = u32::try_from(row).map_err(|_| {
                DodamError::UnsupportedSql(
                    "hash join currently supports up to u32::MAX rows per batch".to_string(),
                )
            })?;
            let row_ref = BuildRowRef {
                batch: batch_index,
                row: row_index,
            };
            if collect_all_rows {
                all_rows.push(row_ref);
            }
            if key_array.is_null(row) {
                continue;
            }
            let value = key_array.value(row);
            min_key = min_key.min(value);
            max_key = max_key.max(value);
            non_null_rows += 1;
        }
    }
    if non_null_rows == 0 {
        return Ok(None);
    }
    let width = usize::try_from(i64::from(max_key) - i64::from(min_key) + 1).ok();
    let Some(width) = width else {
        return Ok(None);
    };
    if width > total_rows.saturating_mul(4).max(1024) {
        return Ok(None);
    }

    let mut dense = vec![None::<Vec<BuildRowRef>>; width];
    let mut has_duplicates = false;
    for (batch_index, batch) in batches.iter().enumerate() {
        let key_index = column_index(batch, key)?;
        let key_array = batch.column(key_index);
        let key_array = key_array
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("Int32 key array");
        for row in 0..batch.num_rows() {
            if key_array.is_null(row) {
                continue;
            }
            let dense_index = (key_array.value(row) - min_key) as usize;
            let row_index = u32::try_from(row).map_err(|_| {
                DodamError::UnsupportedSql(
                    "hash join currently supports up to u32::MAX rows per batch".to_string(),
                )
            })?;
            let row_ref = BuildRowRef {
                batch: batch_index,
                row: row_index,
            };
            let row_refs = dense[dense_index].get_or_insert_with(Vec::new);
            has_duplicates |= !row_refs.is_empty();
            row_refs.push(row_ref);
        }
    }
    let (i32_dense_unique_rows, i32_dense_rows) = if has_duplicates {
        let dense_rows = if width == dense.iter().filter(|row_refs| row_refs.is_some()).count() {
            DenseI32MultiBuildRows::Complete {
                min: min_key,
                rows: dense
                    .into_iter()
                    .map(|row_refs| row_refs.expect("complete dense rows"))
                    .collect(),
            }
        } else {
            DenseI32MultiBuildRows::Sparse {
                min: min_key,
                rows: dense,
            }
        };
        (None, Some(dense_rows))
    } else {
        let dense = dense
            .into_iter()
            .map(|row_refs| row_refs.and_then(|mut row_refs| row_refs.pop()))
            .collect::<Vec<_>>();
        let unique_rows = if width == non_null_rows {
            DenseI32BuildRows::Complete {
                min: min_key,
                rows: dense
                    .into_iter()
                    .map(|row| row.expect("complete dense row"))
                    .collect(),
            }
        } else {
            DenseI32BuildRows::Sparse {
                min: min_key,
                rows: dense,
            }
        };
        (Some(unique_rows), None)
    };

    metrics.add_join_build_rows(total_rows);
    metrics.observe_join_build_bytes(build_bytes);

    Ok(Some(HashJoinBuild {
        key_data_types: vec![DataType::Int32],
        batches: batches.to_vec(),
        all_rows,
        rows: HashMap::new(),
        heavy_rows: HashMap::new(),
        i32_rows: None,
        i32_dense_rows,
        i32_unique_rows: None,
        i32_dense_unique_rows,
        i32_key_set: None,
        i32_dense_key_set: None,
        i32_pair_rows: None,
        i32_pair_unique_rows: None,
        i32_pair_dense_unique_rows: None,
        string_rows: None,
        string_unique_rows: None,
        i64_rows: None,
        i64_unique_rows: None,
        i64_dense_unique_rows: None,
        bloom: BloomFilter::new(total_rows.max(1)),
    }))
}

fn try_build_dense_i32_pair_hash_join_input(
    batches: &[RecordBatch],
    keys: &[String],
    collect_all_rows: bool,
    metrics: &ScanPlanMetricsCounter,
    build_bytes: u64,
) -> Result<Option<HashJoinBuild>> {
    let [left_key, right_key] = keys else {
        return Ok(None);
    };
    let mut min_left = i32::MAX;
    let mut max_left = i32::MIN;
    let mut min_right = i32::MAX;
    let mut max_right = i32::MIN;
    let mut non_null_rows = 0_usize;
    let total_rows = batches.iter().map(RecordBatch::num_rows).sum::<usize>();
    let mut all_rows = collect_all_rows
        .then(|| Vec::with_capacity(total_rows))
        .unwrap_or_default();

    for (batch_index, batch) in batches.iter().enumerate() {
        let left_index = column_index(batch, left_key)?;
        let right_index = column_index(batch, right_key)?;
        let left_array = batch.column(left_index);
        let right_array = batch.column(right_index);
        if left_array.data_type() != &DataType::Int32 || right_array.data_type() != &DataType::Int32
        {
            return Ok(None);
        }
        let left_array = left_array
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("left Int32 key array");
        let right_array = right_array
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("right Int32 key array");
        for row in 0..batch.num_rows() {
            let row_index = u32::try_from(row).map_err(|_| {
                DodamError::UnsupportedSql(
                    "hash join currently supports up to u32::MAX rows per batch".to_string(),
                )
            })?;
            let row_ref = BuildRowRef {
                batch: batch_index,
                row: row_index,
            };
            if collect_all_rows {
                all_rows.push(row_ref);
            }
            if left_array.is_null(row) || right_array.is_null(row) {
                continue;
            }
            let left = left_array.value(row);
            let right = right_array.value(row);
            min_left = min_left.min(left);
            max_left = max_left.max(left);
            min_right = min_right.min(right);
            max_right = max_right.max(right);
            non_null_rows += 1;
        }
    }
    if non_null_rows == 0 {
        return Ok(None);
    }
    let left_width = usize::try_from(i64::from(max_left) - i64::from(min_left) + 1).ok();
    let right_width = usize::try_from(i64::from(max_right) - i64::from(min_right) + 1).ok();
    let Some((left_width, right_width)) = left_width.zip(right_width) else {
        return Ok(None);
    };
    let Some(width) = left_width.checked_mul(right_width) else {
        return Ok(None);
    };
    if width > total_rows.saturating_mul(4).max(1024) {
        return Ok(None);
    }

    let mut dense = vec![None; width];
    for (batch_index, batch) in batches.iter().enumerate() {
        let left_array = batch
            .column(column_index(batch, left_key)?)
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("left Int32 key array");
        let right_array = batch
            .column(column_index(batch, right_key)?)
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("right Int32 key array");
        for row in 0..batch.num_rows() {
            if left_array.is_null(row) || right_array.is_null(row) {
                continue;
            }
            let Some(index) = dense_i32_pair_index(
                left_array.value(row),
                right_array.value(row),
                min_left,
                min_right,
                right_width,
            ) else {
                return Ok(None);
            };
            if dense[index].is_some() {
                return Ok(None);
            }
            let row_index = u32::try_from(row).map_err(|_| {
                DodamError::UnsupportedSql(
                    "hash join currently supports up to u32::MAX rows per batch".to_string(),
                )
            })?;
            dense[index] = Some(BuildRowRef {
                batch: batch_index,
                row: row_index,
            });
        }
    }

    let i32_pair_dense_unique_rows = if width == non_null_rows {
        DenseI32PairBuildRows::Complete {
            min_left,
            min_right,
            right_width,
            rows: dense
                .into_iter()
                .map(|row| row.expect("complete dense pair row"))
                .collect(),
        }
    } else {
        DenseI32PairBuildRows::Sparse {
            min_left,
            min_right,
            right_width,
            rows: dense,
        }
    };

    metrics.add_join_build_rows(total_rows);
    metrics.observe_join_build_bytes(build_bytes);

    Ok(Some(HashJoinBuild {
        key_data_types: vec![DataType::Int32, DataType::Int32],
        batches: batches.to_vec(),
        all_rows,
        rows: HashMap::new(),
        heavy_rows: HashMap::new(),
        i32_rows: None,
        i32_dense_rows: None,
        i32_unique_rows: None,
        i32_dense_unique_rows: None,
        i32_key_set: None,
        i32_dense_key_set: None,
        i32_pair_rows: None,
        i32_pair_unique_rows: None,
        i32_pair_dense_unique_rows: Some(i32_pair_dense_unique_rows),
        string_rows: None,
        string_unique_rows: None,
        i64_rows: None,
        i64_unique_rows: None,
        i64_dense_unique_rows: None,
        bloom: BloomFilter::new(total_rows.max(1)),
    }))
}

fn dense_i32_build_rows(
    rows: &JoinKeyHashMap<i32, BuildRowRef>,
    total_rows: usize,
) -> Option<DenseI32BuildRows> {
    if rows.is_empty() {
        return None;
    }
    let min = *rows.keys().min()?;
    let max = *rows.keys().max()?;
    let width = usize::try_from(i64::from(max) - i64::from(min) + 1).ok()?;
    if width > total_rows.saturating_mul(4).max(1024) {
        return None;
    }
    let mut dense = vec![None; width];
    for (&key, &row_ref) in rows {
        dense[(key - min) as usize] = Some(row_ref);
    }
    if width == rows.len() {
        Some(DenseI32BuildRows::Complete {
            min,
            rows: dense
                .into_iter()
                .map(|row| row.expect("complete dense row"))
                .collect(),
        })
    } else {
        Some(DenseI32BuildRows::Sparse { min, rows: dense })
    }
}

fn dense_i32_multi_build_rows(
    rows: &JoinKeyHashMap<i32, Vec<BuildRowRef>>,
    total_rows: usize,
) -> Option<DenseI32MultiBuildRows> {
    if rows.is_empty() {
        return None;
    }
    let min = *rows.keys().min()?;
    let max = *rows.keys().max()?;
    let width = usize::try_from(i64::from(max) - i64::from(min) + 1).ok()?;
    if width > total_rows.saturating_mul(4).max(1024) {
        return None;
    }
    let mut dense = vec![None; width];
    for (&key, row_refs) in rows {
        dense[(key - min) as usize] = Some(row_refs.clone());
    }
    if width == rows.len() {
        Some(DenseI32MultiBuildRows::Complete {
            min,
            rows: dense
                .into_iter()
                .map(|row_refs| row_refs.expect("complete dense rows"))
                .collect(),
        })
    } else {
        Some(DenseI32MultiBuildRows::Sparse { min, rows: dense })
    }
}

fn dense_i32_key_set(keys: &JoinKeyHashSet<i32>, total_rows: usize) -> Option<DenseI32KeySet> {
    if keys.is_empty() {
        return None;
    }
    let min = *keys.iter().min()?;
    let max = *keys.iter().max()?;
    let width = usize::try_from(i64::from(max) - i64::from(min) + 1).ok()?;
    if width > total_rows.saturating_mul(4).max(1024) {
        return None;
    }
    let mut exists = vec![false; width];
    for &key in keys {
        exists[(key - min) as usize] = true;
    }
    Some(DenseI32KeySet { min, exists })
}

fn dense_i32_pair_build_rows(
    rows: &JoinKeyHashMap<(i32, i32), BuildRowRef>,
    total_rows: usize,
) -> Option<DenseI32PairBuildRows> {
    if rows.is_empty() {
        return None;
    }
    let min_left = rows.keys().map(|(left, _)| *left).min()?;
    let max_left = rows.keys().map(|(left, _)| *left).max()?;
    let min_right = rows.keys().map(|(_, right)| *right).min()?;
    let max_right = rows.keys().map(|(_, right)| *right).max()?;
    let left_width = usize::try_from(i64::from(max_left) - i64::from(min_left) + 1).ok()?;
    let right_width = usize::try_from(i64::from(max_right) - i64::from(min_right) + 1).ok()?;
    let width = left_width.checked_mul(right_width)?;
    if width > total_rows.saturating_mul(4).max(1024) {
        return None;
    }

    let mut dense = vec![None; width];
    for (&(left, right), &row_ref) in rows {
        let index = dense_i32_pair_index(left, right, min_left, min_right, right_width)?;
        dense[index] = Some(row_ref);
    }
    if width == rows.len() {
        Some(DenseI32PairBuildRows::Complete {
            min_left,
            min_right,
            right_width,
            rows: dense
                .into_iter()
                .map(|row| row.expect("complete dense pair row"))
                .collect(),
        })
    } else {
        Some(DenseI32PairBuildRows::Sparse {
            min_left,
            min_right,
            right_width,
            rows: dense,
        })
    }
}

fn dense_i32_pair_index(
    left: i32,
    right: i32,
    min_left: i32,
    min_right: i32,
    right_width: usize,
) -> Option<usize> {
    let left = usize::try_from(left.checked_sub(min_left)?).ok()?;
    let right = usize::try_from(right.checked_sub(min_right)?).ok()?;
    if right >= right_width {
        return None;
    }
    left.checked_mul(right_width)?.checked_add(right)
}

fn insert_i32_build_ref(
    key: i32,
    row_ref: BuildRowRef,
    unique_rows: &mut Option<JoinKeyHashMap<i32, BuildRowRef>>,
    rows: &mut Option<JoinKeyHashMap<i32, Vec<BuildRowRef>>>,
) {
    if let Some(rows) = rows {
        rows.entry(key).or_default().push(row_ref);
        return;
    }

    if unique_rows.is_none() {
        return;
    };
    if let Some(previous) = unique_rows
        .as_mut()
        .expect("unique row map")
        .insert(key, row_ref)
    {
        let unique_map = unique_rows.as_mut().expect("unique row map");
        let mut multi_rows = JoinKeyHashMap::default();
        for (existing_key, existing_ref) in unique_map.drain() {
            multi_rows.insert(existing_key, vec![existing_ref]);
        }
        multi_rows.entry(key).or_default().insert(0, previous);
        *rows = Some(multi_rows);
        *unique_rows = None;
    }
}

fn dense_i64_build_rows(
    rows: &JoinKeyHashMap<i64, BuildRowRef>,
    total_rows: usize,
) -> Option<DenseI64BuildRows> {
    if rows.is_empty() {
        return None;
    }
    let min = *rows.keys().min()?;
    let max = *rows.keys().max()?;
    let width = usize::try_from(max.checked_sub(min)?.checked_add(1)?).ok()?;
    if width > total_rows.saturating_mul(4).max(1024) {
        return None;
    }
    let mut dense = vec![None; width];
    for (&key, &row_ref) in rows {
        dense[usize::try_from(key.checked_sub(min)?).ok()?] = Some(row_ref);
    }
    if width == rows.len() {
        Some(DenseI64BuildRows::Complete {
            min,
            rows: dense
                .into_iter()
                .map(|row| row.expect("complete dense row"))
                .collect(),
        })
    } else {
        Some(DenseI64BuildRows::Sparse { min, rows: dense })
    }
}

fn insert_i64_build_ref(
    key: i64,
    row_ref: BuildRowRef,
    unique_rows: &mut Option<JoinKeyHashMap<i64, BuildRowRef>>,
    rows: &mut Option<JoinKeyHashMap<i64, Vec<BuildRowRef>>>,
) {
    if let Some(rows) = rows {
        rows.entry(key).or_default().push(row_ref);
        return;
    }

    if unique_rows.is_none() {
        return;
    };
    if let Some(previous) = unique_rows
        .as_mut()
        .expect("unique row map")
        .insert(key, row_ref)
    {
        let unique_map = unique_rows.as_mut().expect("unique row map");
        let mut multi_rows = JoinKeyHashMap::default();
        for (existing_key, existing_ref) in unique_map.drain() {
            multi_rows.insert(existing_key, vec![existing_ref]);
        }
        multi_rows.entry(key).or_default().insert(0, previous);
        *rows = Some(multi_rows);
        *unique_rows = None;
    }
}

fn insert_i32_pair_build_ref(
    key: (i32, i32),
    row_ref: BuildRowRef,
    unique_rows: &mut Option<JoinKeyHashMap<(i32, i32), BuildRowRef>>,
    rows: &mut Option<JoinKeyHashMap<(i32, i32), Vec<BuildRowRef>>>,
) {
    if let Some(rows) = rows {
        rows.entry(key).or_default().push(row_ref);
        return;
    }

    if unique_rows.is_none() {
        return;
    };
    if let Some(previous) = unique_rows
        .as_mut()
        .expect("unique pair row map")
        .insert(key, row_ref)
    {
        let unique_map = unique_rows.as_mut().expect("unique pair row map");
        let mut multi_rows = JoinKeyHashMap::default();
        for (existing_key, existing_ref) in unique_map.drain() {
            multi_rows.insert(existing_key, vec![existing_ref]);
        }
        multi_rows.entry(key).or_default().insert(0, previous);
        *rows = Some(multi_rows);
        *unique_rows = None;
    }
}

fn insert_string_build_ref(
    key: &str,
    row_ref: BuildRowRef,
    unique_rows: &mut Option<JoinKeyHashMap<String, BuildRowRef>>,
    rows: &mut Option<JoinKeyHashMap<String, Vec<BuildRowRef>>>,
) {
    if let Some(rows) = rows {
        rows.entry(key.to_string()).or_default().push(row_ref);
        return;
    }

    if unique_rows.is_none() {
        return;
    };
    if let Some(previous) = unique_rows
        .as_mut()
        .expect("unique string row map")
        .insert(key.to_string(), row_ref)
    {
        let unique_map = unique_rows.as_mut().expect("unique string row map");
        let mut multi_rows = JoinKeyHashMap::default();
        for (existing_key, existing_ref) in unique_map.drain() {
            multi_rows.insert(existing_key, vec![existing_ref]);
        }
        multi_rows
            .entry(key.to_string())
            .or_default()
            .insert(0, previous);
        *rows = Some(multi_rows);
        *unique_rows = None;
    }
}

const JOIN_OUTPUT_CHUNK_ROWS: usize = 8192;

struct JoinMaterializeContext<'a> {
    probe: &'a RecordBatch,
    build: &'a RecordBatch,
    left_prefix: &'a str,
    right_prefix: &'a str,
    build_side: JoinBuildSide,
    output_projection: &'a JoinOutputProjection,
    metrics: &'a ScanPlanMetricsCounter,
}

struct HashJoinMaterializeContext<'a> {
    probe: &'a RecordBatch,
    build: &'a HashJoinBuild,
    build_side: JoinBuildSide,
    output_projection: &'a JoinOutputProjection,
    join_output_schema: Arc<Schema>,
    semi_output_schema: Arc<Schema>,
    metrics: &'a ScanPlanMetricsCounter,
}

struct UnmatchedProbeMaterializeContext<'a> {
    build_schema: &'a Schema,
    build_side: JoinBuildSide,
    left_prefix: &'a str,
    right_prefix: &'a str,
    metrics: &'a ScanPlanMetricsCounter,
}

#[allow(clippy::too_many_arguments)]
fn hash_join_materialize_context<'a>(
    probe: &'a RecordBatch,
    build: &'a HashJoinBuild,
    left_prefix: &'a str,
    right_prefix: &'a str,
    build_side: JoinBuildSide,
    output_projection: &'a JoinOutputProjection,
    metrics: &'a ScanPlanMetricsCounter,
) -> Result<HashJoinMaterializeContext<'a>> {
    Ok(HashJoinMaterializeContext {
        probe,
        build,
        build_side,
        output_projection,
        join_output_schema: hash_join_output_schema(
            probe,
            build,
            left_prefix,
            right_prefix,
            build_side,
            output_projection,
        )?,
        semi_output_schema: semi_output_schema(
            probe,
            build,
            build_side,
            left_prefix,
            output_projection.left_columns.as_ref(),
        )?,
        metrics,
    })
}

fn semi_output_schema(
    probe: &RecordBatch,
    build: &HashJoinBuild,
    build_side: JoinBuildSide,
    left_prefix: &str,
    left_columns: Option<&Vec<String>>,
) -> Result<Arc<Schema>> {
    let left_batch = match build_side {
        JoinBuildSide::Left => build.batches.first().ok_or_else(|| {
            DodamError::UnsupportedSql("hash join build input is empty".to_string())
        })?,
        JoinBuildSide::Right => probe,
    };
    single_side_output_schema(left_batch, left_prefix, left_columns)
}

fn hash_join_output_schema(
    probe: &RecordBatch,
    build: &HashJoinBuild,
    left_prefix: &str,
    right_prefix: &str,
    build_side: JoinBuildSide,
    output_projection: &JoinOutputProjection,
) -> Result<Arc<Schema>> {
    let (probe_columns, build_columns) = match build_side {
        JoinBuildSide::Left => (
            output_projection.right_columns.as_ref(),
            output_projection.left_columns.as_ref(),
        ),
        JoinBuildSide::Right => (
            output_projection.left_columns.as_ref(),
            output_projection.right_columns.as_ref(),
        ),
    };
    let build_template = build
        .batches
        .first()
        .ok_or_else(|| DodamError::UnsupportedSql("hash join build input is empty".to_string()))?;
    let probe_fields = projected_qualified_fields(
        probe,
        probe_columns,
        match build_side {
            JoinBuildSide::Left => right_prefix,
            JoinBuildSide::Right => left_prefix,
        },
    )?;
    let build_fields = projected_qualified_fields(
        build_template,
        build_columns,
        match build_side {
            JoinBuildSide::Left => left_prefix,
            JoinBuildSide::Right => right_prefix,
        },
    )?;
    let fields: Vec<Field> = match build_side {
        JoinBuildSide::Left => build_fields.into_iter().chain(probe_fields).collect(),
        JoinBuildSide::Right => probe_fields.into_iter().chain(build_fields).collect(),
    };
    Ok(Arc::new(Schema::new(fields)))
}

fn single_side_output_schema(
    batch: &RecordBatch,
    prefix: &str,
    columns: Option<&Vec<String>>,
) -> Result<Arc<Schema>> {
    Ok(Arc::new(Schema::new(projected_qualified_fields(
        batch, columns, prefix,
    )?)))
}

fn projected_qualified_fields(
    batch: &RecordBatch,
    columns: Option<&Vec<String>>,
    prefix: &str,
) -> Result<Vec<Field>> {
    let fields = batch.schema().fields().clone();
    let indices = match columns {
        Some(columns) => columns
            .iter()
            .map(|column| column_index(batch, column))
            .collect::<Result<Vec<_>>>()?,
        None => (0..fields.len()).collect(),
    };
    Ok(indices
        .into_iter()
        .map(|index| {
            let field = &fields[index];
            Field::new(
                format!("{prefix}.{}", field.name()),
                field.data_type().clone(),
                true,
            )
        })
        .collect())
}

#[allow(clippy::too_many_arguments)]
fn probe_hash_join_batches(
    probe: &RecordBatch,
    build: &HashJoinBuild,
    probe_keys: &[String],
    left_prefix: &str,
    right_prefix: &str,
    build_side: JoinBuildSide,
    join_type: JoinType,
    mut matched_build: Option<&mut MatchedBuildTracker>,
    excluded_probe_keys: Option<&HashSet<OwnedRow>>,
    mut matched_probe_keys: Option<&mut HashSet<OwnedRow>>,
    output_projection: &JoinOutputProjection,
    metrics: &ScanPlanMetricsCounter,
) -> Result<Vec<RecordBatch>> {
    if join_type == JoinType::Semi
        && let Some(i32_dense_key_set) = &build.i32_dense_key_set
        && excluded_probe_keys.is_none()
        && matched_probe_keys.is_none()
    {
        return probe_i32_dense_semi_join_batches(
            probe,
            build,
            i32_dense_key_set,
            probe_keys,
            left_prefix,
            build_side,
            output_projection,
            metrics,
        );
    }

    if join_type == JoinType::Semi
        && let Some(i32_key_set) = &build.i32_key_set
        && excluded_probe_keys.is_none()
        && matched_probe_keys.is_none()
    {
        return probe_i32_semi_join_batches(
            probe,
            build,
            i32_key_set,
            probe_keys,
            left_prefix,
            build_side,
            output_projection,
            metrics,
        );
    }

    if let Some(i32_dense_rows) = &build.i32_dense_unique_rows
        && excluded_probe_keys.is_none()
        && matched_probe_keys.is_none()
    {
        return probe_i32_dense_unique_hash_join_batches(
            probe,
            build,
            i32_dense_rows,
            probe_keys,
            left_prefix,
            right_prefix,
            build_side,
            join_type,
            matched_build,
            output_projection,
            metrics,
        );
    }

    if let Some(i32_unique_rows) = &build.i32_unique_rows
        && excluded_probe_keys.is_none()
        && matched_probe_keys.is_none()
    {
        return probe_i32_unique_hash_join_batches(
            probe,
            build,
            i32_unique_rows,
            probe_keys,
            left_prefix,
            right_prefix,
            build_side,
            join_type,
            matched_build,
            output_projection,
            metrics,
        );
    }

    if let Some(i32_dense_rows) = &build.i32_dense_rows
        && excluded_probe_keys.is_none()
        && matched_probe_keys.is_none()
    {
        return probe_i32_dense_hash_join_batches(
            probe,
            build,
            i32_dense_rows,
            probe_keys,
            left_prefix,
            right_prefix,
            build_side,
            join_type,
            matched_build,
            output_projection,
            metrics,
        );
    }

    if let Some(i32_rows) = &build.i32_rows
        && excluded_probe_keys.is_none()
        && matched_probe_keys.is_none()
    {
        return probe_i32_hash_join_batches(
            probe,
            build,
            i32_rows,
            probe_keys,
            left_prefix,
            right_prefix,
            build_side,
            join_type,
            matched_build,
            output_projection,
            metrics,
        );
    }

    if let Some(i64_dense_rows) = &build.i64_dense_unique_rows
        && excluded_probe_keys.is_none()
        && matched_probe_keys.is_none()
    {
        return probe_i64_dense_unique_hash_join_batches(
            probe,
            build,
            i64_dense_rows,
            probe_keys,
            left_prefix,
            right_prefix,
            build_side,
            join_type,
            matched_build,
            output_projection,
            metrics,
        );
    }

    if let Some(i64_unique_rows) = &build.i64_unique_rows
        && excluded_probe_keys.is_none()
        && matched_probe_keys.is_none()
    {
        return probe_i64_unique_hash_join_batches(
            probe,
            build,
            i64_unique_rows,
            probe_keys,
            left_prefix,
            right_prefix,
            build_side,
            join_type,
            matched_build,
            output_projection,
            metrics,
        );
    }

    if let Some(i64_rows) = &build.i64_rows
        && excluded_probe_keys.is_none()
        && matched_probe_keys.is_none()
    {
        return probe_i64_hash_join_batches(
            probe,
            build,
            i64_rows,
            probe_keys,
            left_prefix,
            right_prefix,
            build_side,
            join_type,
            matched_build,
            output_projection,
            metrics,
        );
    }

    if let Some(i32_pair_dense_unique_rows) = &build.i32_pair_dense_unique_rows
        && excluded_probe_keys.is_none()
        && matched_probe_keys.is_none()
    {
        return probe_i32_pair_dense_unique_hash_join_batches(
            probe,
            build,
            i32_pair_dense_unique_rows,
            probe_keys,
            left_prefix,
            right_prefix,
            build_side,
            join_type,
            matched_build,
            output_projection,
            metrics,
        );
    }

    if let Some(i32_pair_unique_rows) = &build.i32_pair_unique_rows
        && excluded_probe_keys.is_none()
        && matched_probe_keys.is_none()
    {
        return probe_i32_pair_unique_hash_join_batches(
            probe,
            build,
            i32_pair_unique_rows,
            probe_keys,
            left_prefix,
            right_prefix,
            build_side,
            join_type,
            matched_build,
            output_projection,
            metrics,
        );
    }

    if let Some(i32_pair_rows) = &build.i32_pair_rows
        && excluded_probe_keys.is_none()
        && matched_probe_keys.is_none()
    {
        return probe_i32_pair_hash_join_batches(
            probe,
            build,
            i32_pair_rows,
            probe_keys,
            left_prefix,
            right_prefix,
            build_side,
            join_type,
            matched_build,
            output_projection,
            metrics,
        );
    }

    if let Some(string_unique_rows) = &build.string_unique_rows
        && excluded_probe_keys.is_none()
        && matched_probe_keys.is_none()
    {
        return probe_string_unique_hash_join_batches(
            probe,
            build,
            string_unique_rows,
            probe_keys,
            left_prefix,
            right_prefix,
            build_side,
            join_type,
            matched_build,
            output_projection,
            metrics,
        );
    }

    if let Some(string_rows) = &build.string_rows
        && excluded_probe_keys.is_none()
        && matched_probe_keys.is_none()
    {
        return probe_string_hash_join_batches(
            probe,
            build,
            string_rows,
            probe_keys,
            left_prefix,
            right_prefix,
            build_side,
            join_type,
            matched_build,
            output_projection,
            metrics,
        );
    }

    let probe_key_arrays = key_arrays(probe, probe_keys)?;
    let probe_key_types = probe_key_arrays
        .iter()
        .map(|array| array.data_type().clone())
        .collect::<Vec<_>>();
    if probe_key_types != build.key_data_types {
        return Err(DodamError::UnsupportedSql(format!(
            "JOIN key types must match: probe side is {:?}, build side is {:?}",
            probe_key_types, build.key_data_types
        )));
    }

    let probe_converter = RowConverter::new(
        probe_key_arrays
            .iter()
            .map(|array| SortField::new(array.data_type().clone()))
            .collect(),
    )?;
    let probe_rows = probe_converter.convert_columns(&probe_key_arrays)?;
    let mut probe_indices = Vec::new();
    let mut build_refs = Vec::new();
    let mut semi_indices = Vec::new();
    let mut output = Vec::new();
    let mut bloom_filtered_rows = 0_usize;
    let context = hash_join_materialize_context(
        probe,
        build,
        left_prefix,
        right_prefix,
        build_side,
        output_projection,
        metrics,
    )?;
    metrics.add_join_probe_rows(probe.num_rows());

    for (probe_row, key) in probe_rows.iter().enumerate() {
        if probe_key_arrays
            .iter()
            .any(|array| array.is_null(probe_row))
        {
            push_unmatched_probe_if_outer(
                probe,
                build,
                probe_row,
                build_side,
                join_type,
                left_prefix,
                right_prefix,
                metrics,
                &mut output,
            )?;
            continue;
        }
        let key = key.owned();
        if excluded_probe_keys.is_some_and(|excluded| excluded.contains(&key)) {
            continue;
        }
        if !build.bloom.might_contain(&key) {
            bloom_filtered_rows += 1;
            push_unmatched_probe_if_outer(
                probe,
                build,
                probe_row,
                build_side,
                join_type,
                left_prefix,
                right_prefix,
                metrics,
                &mut output,
            )?;
            continue;
        }
        let mut matched = false;
        if let Some(matches) = build.rows.get(&key) {
            mark_probe_match(&mut matched_probe_keys, &key);
            mark_build_matches(&mut matched_build, matches);
            let probe_row = u32::try_from(probe_row).map_err(|_| {
                DodamError::UnsupportedSql(
                    "hash join currently supports up to u32::MAX rows per side".to_string(),
                )
            })?;
            if join_type == JoinType::Semi {
                push_semi_join_match(&context, probe_row, &mut semi_indices, &mut output)?;
                continue;
            }
            push_hash_join_matches(
                &context,
                probe_row,
                matches,
                &mut probe_indices,
                &mut build_refs,
                &mut output,
            )?;
            continue;
        }
        if let Some(matches) = build.heavy_rows.get(&key) {
            matched = true;
            mark_probe_match(&mut matched_probe_keys, &key);
            mark_build_matches(&mut matched_build, matches);
            let probe_row = u32::try_from(probe_row).map_err(|_| {
                DodamError::UnsupportedSql(
                    "hash join currently supports up to u32::MAX rows per side".to_string(),
                )
            })?;
            if join_type == JoinType::Semi {
                push_semi_join_match(&context, probe_row, &mut semi_indices, &mut output)?;
                continue;
            }
            push_hash_join_matches(
                &context,
                probe_row,
                matches,
                &mut probe_indices,
                &mut build_refs,
                &mut output,
            )?;
        }
        if !matched {
            push_unmatched_probe_if_outer(
                probe,
                build,
                probe_row,
                build_side,
                join_type,
                left_prefix,
                right_prefix,
                metrics,
                &mut output,
            )?;
        }
    }
    metrics.add_join_bloom_filtered_rows(bloom_filtered_rows);

    if !probe_indices.is_empty() {
        output.push(materialize_hash_join_pairs(
            &context,
            &probe_indices,
            &build_refs,
        )?);
    }
    if !semi_indices.is_empty() {
        output.push(materialize_semi_join_rows(&context, &semi_indices)?);
    }

    Ok(output)
}

fn discard_hash_join_batches(
    probe: &RecordBatch,
    build: &HashJoinBuild,
    probe_keys: &[String],
    join_type: JoinType,
    excluded_probe_keys: Option<&HashSet<OwnedRow>>,
    metrics: &ScanPlanMetricsCounter,
) -> Result<()> {
    metrics.add_join_probe_rows(probe.num_rows());
    let _ = (build, probe_keys, join_type, excluded_probe_keys);
    Ok(())
}

fn try_probe_i32_semi_join_to_i32_sink(
    probe: &RecordBatch,
    build: &HashJoinBuild,
    probe_keys: &[String],
    join_type: JoinType,
    output_projection: &JoinOutputProjection,
    metrics: &ScanPlanMetricsCounter,
    sink: &mut dyn RecordBatchSink,
) -> Result<bool> {
    if !sink.supports_i32_rows() || join_type != JoinType::Semi {
        return Ok(false);
    }
    if output_projection.left_columns.as_deref() != Some(&["id".to_string()])
        || output_projection
            .right_columns
            .as_ref()
            .is_some_and(|columns| !columns.is_empty())
    {
        return Ok(false);
    }
    let [probe_key] = probe_keys else {
        return Ok(false);
    };
    let key_array = probe.column(column_index(probe, probe_key)?);
    if key_array.data_type() != &DataType::Int32 || build.key_data_types != [DataType::Int32] {
        return Ok(false);
    }
    let key_array = key_array
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("Int32 probe key array");
    let id_array = probe.column(column_index(probe, "id")?);
    let Some(id_array) = id_array.as_any().downcast_ref::<Int32Array>() else {
        return Ok(false);
    };

    if let Some(i32_key_set) = &build.i32_dense_key_set {
        return write_i32_dense_semi_join_to_i32_sink(
            probe,
            key_array,
            i32_key_set,
            id_array,
            metrics,
            sink,
        );
    }
    if let Some(i32_key_set) = &build.i32_key_set {
        return write_i32_semi_join_to_i32_sink(
            probe,
            key_array,
            i32_key_set,
            id_array,
            metrics,
            sink,
        );
    }
    Ok(false)
}

fn write_i32_dense_semi_join_to_i32_sink(
    probe: &RecordBatch,
    key_array: &Int32Array,
    key_set: &DenseI32KeySet,
    id_array: &Int32Array,
    metrics: &ScanPlanMetricsCounter,
    sink: &mut dyn RecordBatchSink,
) -> Result<bool> {
    let mut indices = Vec::with_capacity(JOIN_OUTPUT_CHUNK_ROWS);
    let mut bloom_filtered_rows = 0_usize;
    let mut output_rows = 0_usize;
    let started = Instant::now();
    metrics.add_join_probe_rows(probe.num_rows());
    for probe_row in 0..probe.num_rows() {
        if key_array.is_null(probe_row) {
            continue;
        }
        if !key_set.contains(key_array.value(probe_row)) {
            bloom_filtered_rows += 1;
            continue;
        }
        push_direct_i32_row(probe_row, id_array, &mut indices, &mut output_rows, sink)?;
    }
    flush_direct_i32_rows(id_array, &mut indices, &mut output_rows, sink)?;
    metrics.add_join_bloom_filtered_rows(bloom_filtered_rows);
    metrics.add_join_output_rows(output_rows);
    metrics.add_join_materialize_time(started.elapsed());
    Ok(true)
}

fn write_i32_semi_join_to_i32_sink(
    probe: &RecordBatch,
    key_array: &Int32Array,
    key_set: &JoinKeyHashSet<i32>,
    id_array: &Int32Array,
    metrics: &ScanPlanMetricsCounter,
    sink: &mut dyn RecordBatchSink,
) -> Result<bool> {
    let mut indices = Vec::with_capacity(JOIN_OUTPUT_CHUNK_ROWS);
    let mut bloom_filtered_rows = 0_usize;
    let mut output_rows = 0_usize;
    let started = Instant::now();
    metrics.add_join_probe_rows(probe.num_rows());
    for probe_row in 0..probe.num_rows() {
        if key_array.is_null(probe_row) {
            continue;
        }
        if !key_set.contains(&key_array.value(probe_row)) {
            bloom_filtered_rows += 1;
            continue;
        }
        push_direct_i32_row(probe_row, id_array, &mut indices, &mut output_rows, sink)?;
    }
    flush_direct_i32_rows(id_array, &mut indices, &mut output_rows, sink)?;
    metrics.add_join_bloom_filtered_rows(bloom_filtered_rows);
    metrics.add_join_output_rows(output_rows);
    metrics.add_join_materialize_time(started.elapsed());
    Ok(true)
}

fn push_direct_i32_row(
    probe_row: usize,
    id_array: &Int32Array,
    indices: &mut Vec<u32>,
    output_rows: &mut usize,
    sink: &mut dyn RecordBatchSink,
) -> Result<()> {
    let probe_row = u32::try_from(probe_row).map_err(|_| {
        DodamError::UnsupportedSql(
            "hash join currently supports up to u32::MAX rows per side".to_string(),
        )
    })?;
    indices.push(probe_row);
    if indices.len() >= JOIN_OUTPUT_CHUNK_ROWS {
        flush_direct_i32_rows(id_array, indices, output_rows, sink)?;
    }
    Ok(())
}

fn flush_direct_i32_rows(
    id_array: &Int32Array,
    indices: &mut Vec<u32>,
    output_rows: &mut usize,
    sink: &mut dyn RecordBatchSink,
) -> Result<()> {
    if indices.is_empty() {
        return Ok(());
    }
    if !sink.write_i32_rows(id_array, indices)? {
        return Err(DodamError::UnsupportedSql(
            "direct join sink rejected supported Int32 rows".to_string(),
        ));
    }
    *output_rows = output_rows.saturating_add(indices.len());
    indices.clear();
    Ok(())
}

fn try_probe_i32_dense_join_to_i32_utf8_sink(
    probe: &RecordBatch,
    build: &HashJoinBuild,
    probe_keys: &[String],
    join_type: JoinType,
    output_projection: &JoinOutputProjection,
    metrics: &ScanPlanMetricsCounter,
    sink: &mut dyn RecordBatchSink,
) -> Result<bool> {
    if !sink.supports_i32_utf8_rows() {
        return Ok(false);
    }
    if join_type != JoinType::Inner {
        return Ok(false);
    }
    if output_projection.left_columns.as_deref() != Some(&["id".to_string()])
        || output_projection.right_columns.as_deref() != Some(&["payload".to_string()])
    {
        return Ok(false);
    }
    let Some(i32_rows) = &build.i32_dense_rows else {
        return Ok(false);
    };
    let [probe_key] = probe_keys else {
        return Ok(false);
    };
    let key_array = probe.column(column_index(probe, probe_key)?);
    if key_array.data_type() != &DataType::Int32 || build.key_data_types != [DataType::Int32] {
        return Ok(false);
    }
    let key_array = key_array
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("Int32 probe key array");
    let id_array = probe.column(column_index(probe, "id")?);
    let Some(id_array) = id_array.as_any().downcast_ref::<Int32Array>() else {
        return Ok(false);
    };
    let payload_index = column_index(&build.batches[0], "payload")?;
    let payload_arrays = build_column_arrays::<StringArray>(build, payload_index)?;

    let mut probe_indices = Vec::with_capacity(JOIN_OUTPUT_CHUNK_ROWS);
    let mut build_batch_indices = Vec::with_capacity(JOIN_OUTPUT_CHUNK_ROWS);
    let mut build_row_indices = Vec::with_capacity(JOIN_OUTPUT_CHUNK_ROWS);
    let mut bloom_filtered_rows = 0_usize;
    let mut output_rows = 0_usize;
    let started = Instant::now();
    metrics.add_join_probe_rows(probe.num_rows());

    for probe_row in 0..probe.num_rows() {
        if key_array.is_null(probe_row) {
            continue;
        }
        let Some(matches) = i32_rows.get(key_array.value(probe_row)) else {
            bloom_filtered_rows += 1;
            continue;
        };
        let probe_row = u32::try_from(probe_row).map_err(|_| {
            DodamError::UnsupportedSql(
                "hash join currently supports up to u32::MAX rows per side".to_string(),
            )
        })?;
        for build_ref in matches {
            probe_indices.push(probe_row);
            build_batch_indices.push(build_ref.batch);
            build_row_indices.push(build_ref.row);
            if probe_indices.len() >= JOIN_OUTPUT_CHUNK_ROWS {
                if !sink.write_i32_utf8_rows(
                    id_array,
                    &probe_indices,
                    &payload_arrays,
                    &build_batch_indices,
                    &build_row_indices,
                )? {
                    return Ok(false);
                }
                output_rows = output_rows.saturating_add(probe_indices.len());
                probe_indices.clear();
                build_batch_indices.clear();
                build_row_indices.clear();
            }
        }
    }

    if !probe_indices.is_empty() {
        if !sink.write_i32_utf8_rows(
            id_array,
            &probe_indices,
            &payload_arrays,
            &build_batch_indices,
            &build_row_indices,
        )? {
            return Ok(false);
        }
        output_rows = output_rows.saturating_add(probe_indices.len());
    }
    metrics.add_join_bloom_filtered_rows(bloom_filtered_rows);
    metrics.add_join_output_rows(output_rows);
    metrics.add_join_materialize_time(started.elapsed());
    Ok(true)
}

fn try_probe_unique_join_to_i32_utf8_sink(
    probe: &RecordBatch,
    build: &HashJoinBuild,
    probe_keys: &[String],
    join_type: JoinType,
    output_projection: &JoinOutputProjection,
    metrics: &ScanPlanMetricsCounter,
    sink: &mut dyn RecordBatchSink,
) -> Result<bool> {
    if !sink.supports_i32_utf8_rows() || join_type != JoinType::Inner {
        return Ok(false);
    }
    if output_projection.left_columns.as_deref() != Some(&["id".to_string()])
        || output_projection.right_columns.as_deref() != Some(&["payload".to_string()])
    {
        return Ok(false);
    }
    let id_array = probe.column(column_index(probe, "id")?);
    let Some(id_array) = id_array.as_any().downcast_ref::<Int32Array>() else {
        return Ok(false);
    };
    let payload_index = column_index(&build.batches[0], "payload")?;
    let payload_arrays = build_column_arrays::<StringArray>(build, payload_index)?;

    if let Some(i32_rows) = &build.i32_unique_rows {
        let [probe_key] = probe_keys else {
            return Ok(false);
        };
        let key_array = probe.column(column_index(probe, probe_key)?);
        if key_array.data_type() != &DataType::Int32 || build.key_data_types != [DataType::Int32] {
            return Ok(false);
        }
        let key_array = key_array
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("Int32 probe key array");
        return write_i32_unique_join_to_i32_utf8_sink(
            probe,
            key_array,
            i32_rows,
            id_array,
            &payload_arrays,
            metrics,
            sink,
        );
    }

    if let Some(string_rows) = &build.string_unique_rows {
        let [probe_key] = probe_keys else {
            return Ok(false);
        };
        let key_array = probe.column(column_index(probe, probe_key)?);
        if key_array.data_type() != &DataType::Utf8 || build.key_data_types != [DataType::Utf8] {
            return Ok(false);
        }
        let key_array = key_array
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("Utf8 probe key array");
        return write_string_unique_join_to_i32_utf8_sink(
            probe,
            key_array,
            string_rows,
            id_array,
            &payload_arrays,
            metrics,
            sink,
        );
    }

    Ok(false)
}

fn write_i32_unique_join_to_i32_utf8_sink(
    probe: &RecordBatch,
    key_array: &Int32Array,
    rows: &JoinKeyHashMap<i32, BuildRowRef>,
    id_array: &Int32Array,
    payload_arrays: &[&StringArray],
    metrics: &ScanPlanMetricsCounter,
    sink: &mut dyn RecordBatchSink,
) -> Result<bool> {
    let mut probe_indices = Vec::with_capacity(JOIN_OUTPUT_CHUNK_ROWS);
    let mut build_batch_indices = Vec::with_capacity(JOIN_OUTPUT_CHUNK_ROWS);
    let mut build_row_indices = Vec::with_capacity(JOIN_OUTPUT_CHUNK_ROWS);
    let mut bloom_filtered_rows = 0_usize;
    let mut output_rows = 0_usize;
    let started = Instant::now();
    metrics.add_join_probe_rows(probe.num_rows());
    for probe_row in 0..probe.num_rows() {
        if key_array.is_null(probe_row) {
            continue;
        }
        let Some(build_ref) = rows.get(&key_array.value(probe_row)).copied() else {
            bloom_filtered_rows += 1;
            continue;
        };
        push_direct_i32_utf8_row(
            probe_row,
            build_ref,
            id_array,
            payload_arrays,
            &mut probe_indices,
            &mut build_batch_indices,
            &mut build_row_indices,
            &mut output_rows,
            sink,
        )?;
    }
    flush_direct_i32_utf8_rows(
        id_array,
        payload_arrays,
        &mut probe_indices,
        &mut build_batch_indices,
        &mut build_row_indices,
        &mut output_rows,
        sink,
    )?;
    metrics.add_join_bloom_filtered_rows(bloom_filtered_rows);
    metrics.add_join_output_rows(output_rows);
    metrics.add_join_materialize_time(started.elapsed());
    Ok(true)
}

fn write_string_unique_join_to_i32_utf8_sink(
    probe: &RecordBatch,
    key_array: &StringArray,
    rows: &JoinKeyHashMap<String, BuildRowRef>,
    id_array: &Int32Array,
    payload_arrays: &[&StringArray],
    metrics: &ScanPlanMetricsCounter,
    sink: &mut dyn RecordBatchSink,
) -> Result<bool> {
    let mut probe_indices = Vec::with_capacity(JOIN_OUTPUT_CHUNK_ROWS);
    let mut build_batch_indices = Vec::with_capacity(JOIN_OUTPUT_CHUNK_ROWS);
    let mut build_row_indices = Vec::with_capacity(JOIN_OUTPUT_CHUNK_ROWS);
    let mut bloom_filtered_rows = 0_usize;
    let mut output_rows = 0_usize;
    let started = Instant::now();
    metrics.add_join_probe_rows(probe.num_rows());
    for probe_row in 0..probe.num_rows() {
        if key_array.is_null(probe_row) {
            continue;
        }
        let Some(build_ref) = rows.get(key_array.value(probe_row)).copied() else {
            bloom_filtered_rows += 1;
            continue;
        };
        push_direct_i32_utf8_row(
            probe_row,
            build_ref,
            id_array,
            payload_arrays,
            &mut probe_indices,
            &mut build_batch_indices,
            &mut build_row_indices,
            &mut output_rows,
            sink,
        )?;
    }
    flush_direct_i32_utf8_rows(
        id_array,
        payload_arrays,
        &mut probe_indices,
        &mut build_batch_indices,
        &mut build_row_indices,
        &mut output_rows,
        sink,
    )?;
    metrics.add_join_bloom_filtered_rows(bloom_filtered_rows);
    metrics.add_join_output_rows(output_rows);
    metrics.add_join_materialize_time(started.elapsed());
    Ok(true)
}

#[allow(clippy::too_many_arguments)]
fn push_direct_i32_utf8_row(
    probe_row: usize,
    build_ref: BuildRowRef,
    id_array: &Int32Array,
    payload_arrays: &[&StringArray],
    probe_indices: &mut Vec<u32>,
    build_batch_indices: &mut Vec<usize>,
    build_row_indices: &mut Vec<u32>,
    output_rows: &mut usize,
    sink: &mut dyn RecordBatchSink,
) -> Result<()> {
    let probe_row = u32::try_from(probe_row).map_err(|_| {
        DodamError::UnsupportedSql(
            "hash join currently supports up to u32::MAX rows per side".to_string(),
        )
    })?;
    probe_indices.push(probe_row);
    build_batch_indices.push(build_ref.batch);
    build_row_indices.push(build_ref.row);
    if probe_indices.len() >= JOIN_OUTPUT_CHUNK_ROWS {
        flush_direct_i32_utf8_rows(
            id_array,
            payload_arrays,
            probe_indices,
            build_batch_indices,
            build_row_indices,
            output_rows,
            sink,
        )?;
    }
    Ok(())
}

fn flush_direct_i32_utf8_rows(
    id_array: &Int32Array,
    payload_arrays: &[&StringArray],
    probe_indices: &mut Vec<u32>,
    build_batch_indices: &mut Vec<usize>,
    build_row_indices: &mut Vec<u32>,
    output_rows: &mut usize,
    sink: &mut dyn RecordBatchSink,
) -> Result<()> {
    if probe_indices.is_empty() {
        return Ok(());
    }
    if !sink.write_i32_utf8_rows(
        id_array,
        probe_indices,
        payload_arrays,
        build_batch_indices,
        build_row_indices,
    )? {
        return Err(DodamError::UnsupportedSql(
            "direct join sink rejected supported rows".to_string(),
        ));
    }
    *output_rows = output_rows.saturating_add(probe_indices.len());
    probe_indices.clear();
    build_batch_indices.clear();
    build_row_indices.clear();
    Ok(())
}

#[allow(dead_code)]
fn count_hash_join_output_rows(
    probe: &RecordBatch,
    build: &HashJoinBuild,
    probe_keys: &[String],
    join_type: JoinType,
    excluded_probe_keys: Option<&HashSet<OwnedRow>>,
) -> Result<usize> {
    if join_type == JoinType::Semi
        && let Some(i32_dense_key_set) = &build.i32_dense_key_set
    {
        let key_array = single_i32_probe_key_array(probe, probe_keys, &build.key_data_types)?;
        let mut rows = 0_usize;
        for probe_row in 0..probe.num_rows() {
            if !key_array.is_null(probe_row)
                && i32_dense_key_set.contains(key_array.value(probe_row))
            {
                rows += 1;
            }
        }
        return Ok(rows);
    }
    if let Some(i32_dense_rows) = &build.i32_dense_rows {
        let key_array = single_i32_probe_key_array(probe, probe_keys, &build.key_data_types)?;
        let mut rows = 0_usize;
        for probe_row in 0..probe.num_rows() {
            if key_array.is_null(probe_row) {
                continue;
            }
            if let Some(matches) = i32_dense_rows.get(key_array.value(probe_row)) {
                rows += if join_type == JoinType::Semi {
                    1
                } else {
                    matches.len()
                };
            }
        }
        return Ok(rows);
    }
    if let Some(i32_dense_rows) = &build.i32_dense_unique_rows {
        let key_array = single_i32_probe_key_array(probe, probe_keys, &build.key_data_types)?;
        let mut rows = 0_usize;
        for probe_row in 0..probe.num_rows() {
            if !key_array.is_null(probe_row)
                && i32_dense_rows.get(key_array.value(probe_row)).is_some()
            {
                rows += 1;
            }
        }
        return Ok(rows);
    }
    if let Some(i32_pair_rows) = &build.i32_pair_rows {
        let [left_key, right_key] = probe_keys else {
            return Err(DodamError::UnsupportedSql(
                "Int32 pair hash join fast path expects exactly two probe keys".to_string(),
            ));
        };
        let left_key_array = probe
            .column(column_index(probe, left_key)?)
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("first Int32 probe key array");
        let right_key_array = probe
            .column(column_index(probe, right_key)?)
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("second Int32 probe key array");
        let mut rows = 0_usize;
        for probe_row in 0..probe.num_rows() {
            if left_key_array.is_null(probe_row) || right_key_array.is_null(probe_row) {
                continue;
            }
            if let Some(matches) = i32_pair_rows.get(&(
                left_key_array.value(probe_row),
                right_key_array.value(probe_row),
            )) {
                rows += if join_type == JoinType::Semi {
                    1
                } else {
                    matches.len()
                };
            }
        }
        return Ok(rows);
    }
    if let Some(string_rows) = &build.string_rows {
        let [probe_key] = probe_keys else {
            return Err(DodamError::UnsupportedSql(
                "String hash join fast path expects exactly one probe key".to_string(),
            ));
        };
        let key_array = probe
            .column(column_index(probe, probe_key)?)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("Utf8 probe key array");
        let mut rows = 0_usize;
        for probe_row in 0..probe.num_rows() {
            if key_array.is_null(probe_row) {
                continue;
            }
            if let Some(matches) = string_rows.get(key_array.value(probe_row)) {
                rows += if join_type == JoinType::Semi {
                    1
                } else {
                    matches.len()
                };
            }
        }
        return Ok(rows);
    }

    let probe_key_arrays = key_arrays(probe, probe_keys)?;
    let probe_converter = RowConverter::new(
        probe_key_arrays
            .iter()
            .map(|array| SortField::new(array.data_type().clone()))
            .collect(),
    )?;
    let probe_rows = probe_converter.convert_columns(&probe_key_arrays)?;
    let mut rows = 0_usize;
    for (probe_row, key) in probe_rows.iter().enumerate() {
        if probe_key_arrays
            .iter()
            .any(|array| array.is_null(probe_row))
        {
            continue;
        }
        let key = key.owned();
        if excluded_probe_keys.is_some_and(|excluded| excluded.contains(&key)) {
            continue;
        }
        if let Some(matches) = build.rows.get(&key).or_else(|| build.heavy_rows.get(&key)) {
            rows += if join_type == JoinType::Semi {
                1
            } else {
                matches.len()
            };
        }
    }
    Ok(rows)
}

#[allow(dead_code)]
fn single_i32_probe_key_array<'a>(
    probe: &'a RecordBatch,
    probe_keys: &[String],
    build_key_data_types: &[DataType],
) -> Result<&'a Int32Array> {
    let [probe_key] = probe_keys else {
        return Err(DodamError::UnsupportedSql(
            "Int32 hash join fast path expects exactly one probe key".to_string(),
        ));
    };
    let key_array = probe.column(column_index(probe, probe_key)?);
    if key_array.data_type() != &DataType::Int32 || build_key_data_types != [DataType::Int32] {
        return Err(DodamError::UnsupportedSql(format!(
            "JOIN key types must match: probe side is {:?}, build side is {:?}",
            vec![key_array.data_type().clone()],
            build_key_data_types
        )));
    }
    Ok(key_array
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("Int32 probe key array"))
}

#[allow(clippy::too_many_arguments)]
fn probe_i32_dense_semi_join_batches(
    probe: &RecordBatch,
    build: &HashJoinBuild,
    i32_key_set: &DenseI32KeySet,
    probe_keys: &[String],
    left_prefix: &str,
    build_side: JoinBuildSide,
    output_projection: &JoinOutputProjection,
    metrics: &ScanPlanMetricsCounter,
) -> Result<Vec<RecordBatch>> {
    if build_side != JoinBuildSide::Right {
        return Err(DodamError::UnsupportedSql(
            "SEMI JOIN currently expects the right side as hash build input".to_string(),
        ));
    }
    let [probe_key] = probe_keys else {
        return Err(DodamError::UnsupportedSql(
            "Int32 SEMI JOIN fast path expects exactly one probe key".to_string(),
        ));
    };
    let key_index = column_index(probe, probe_key)?;
    let key_array = probe.column(key_index);
    if key_array.data_type() != &DataType::Int32 {
        return Err(DodamError::UnsupportedSql(format!(
            "JOIN key types must match: probe side is {:?}, build side is {:?}",
            vec![key_array.data_type().clone()],
            build.key_data_types
        )));
    }
    let key_array = key_array
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("Int32 probe key array");
    let context = hash_join_materialize_context(
        probe,
        build,
        left_prefix,
        "",
        build_side,
        output_projection,
        metrics,
    )?;
    let mut output = Vec::new();
    let mut semi_indices = Vec::with_capacity(probe.num_rows().min(JOIN_OUTPUT_CHUNK_ROWS));
    let mut bloom_filtered_rows = 0_usize;
    metrics.add_join_probe_rows(probe.num_rows());

    for probe_row in 0..probe.num_rows() {
        if key_array.is_null(probe_row) {
            continue;
        }
        if !i32_key_set.contains(key_array.value(probe_row)) {
            bloom_filtered_rows += 1;
            continue;
        }
        let probe_row = u32::try_from(probe_row).map_err(|_| {
            DodamError::UnsupportedSql(
                "hash join currently supports up to u32::MAX rows per side".to_string(),
            )
        })?;
        semi_indices.push(probe_row);
        if semi_indices.len() >= JOIN_OUTPUT_CHUNK_ROWS {
            output.push(materialize_semi_join_rows(&context, &semi_indices)?);
            semi_indices.clear();
        }
    }
    metrics.add_join_bloom_filtered_rows(bloom_filtered_rows);
    if !semi_indices.is_empty() {
        output.push(materialize_semi_join_rows(&context, &semi_indices)?);
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn probe_i32_semi_join_batches(
    probe: &RecordBatch,
    build: &HashJoinBuild,
    i32_key_set: &JoinKeyHashSet<i32>,
    probe_keys: &[String],
    left_prefix: &str,
    build_side: JoinBuildSide,
    output_projection: &JoinOutputProjection,
    metrics: &ScanPlanMetricsCounter,
) -> Result<Vec<RecordBatch>> {
    if build_side != JoinBuildSide::Right {
        return Err(DodamError::UnsupportedSql(
            "SEMI JOIN currently expects the right side as hash build input".to_string(),
        ));
    }
    let [probe_key] = probe_keys else {
        return Err(DodamError::UnsupportedSql(
            "Int32 SEMI JOIN fast path expects exactly one probe key".to_string(),
        ));
    };
    let key_index = column_index(probe, probe_key)?;
    let key_array = probe.column(key_index);
    if key_array.data_type() != &DataType::Int32 {
        return Err(DodamError::UnsupportedSql(format!(
            "JOIN key types must match: probe side is {:?}, build side is {:?}",
            vec![key_array.data_type().clone()],
            build.key_data_types
        )));
    }
    let key_array = key_array
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("Int32 probe key array");
    let context = hash_join_materialize_context(
        probe,
        build,
        left_prefix,
        "",
        build_side,
        output_projection,
        metrics,
    )?;
    let mut output = Vec::new();
    let mut semi_indices = Vec::new();
    let mut bloom_filtered_rows = 0_usize;
    metrics.add_join_probe_rows(probe.num_rows());

    for probe_row in 0..probe.num_rows() {
        if key_array.is_null(probe_row) {
            continue;
        }
        if !i32_key_set.contains(&key_array.value(probe_row)) {
            bloom_filtered_rows += 1;
            continue;
        }
        let probe_row = u32::try_from(probe_row).map_err(|_| {
            DodamError::UnsupportedSql(
                "hash join currently supports up to u32::MAX rows per side".to_string(),
            )
        })?;
        semi_indices.push(probe_row);
        if semi_indices.len() >= JOIN_OUTPUT_CHUNK_ROWS {
            output.push(materialize_semi_join_rows(&context, &semi_indices)?);
            semi_indices.clear();
        }
    }
    metrics.add_join_bloom_filtered_rows(bloom_filtered_rows);
    if !semi_indices.is_empty() {
        output.push(materialize_semi_join_rows(&context, &semi_indices)?);
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn probe_i32_dense_unique_hash_join_batches(
    probe: &RecordBatch,
    build: &HashJoinBuild,
    i32_dense_rows: &DenseI32BuildRows,
    probe_keys: &[String],
    left_prefix: &str,
    right_prefix: &str,
    build_side: JoinBuildSide,
    join_type: JoinType,
    mut matched_build: Option<&mut MatchedBuildTracker>,
    output_projection: &JoinOutputProjection,
    metrics: &ScanPlanMetricsCounter,
) -> Result<Vec<RecordBatch>> {
    let [probe_key] = probe_keys else {
        return Err(DodamError::UnsupportedSql(
            "Int32 dense hash join fast path expects exactly one probe key".to_string(),
        ));
    };
    let key_index = column_index(probe, probe_key)?;
    let key_array = probe.column(key_index);
    if key_array.data_type() != &DataType::Int32 {
        return Err(DodamError::UnsupportedSql(format!(
            "JOIN key types must match: probe side is {:?}, build side is {:?}",
            vec![key_array.data_type().clone()],
            build.key_data_types
        )));
    }
    let key_array = key_array
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("Int32 probe key array");
    let mut probe_indices = Vec::with_capacity(probe.num_rows().min(JOIN_OUTPUT_CHUNK_ROWS));
    let mut build_refs = Vec::with_capacity(probe.num_rows().min(JOIN_OUTPUT_CHUNK_ROWS));
    let mut unmatched_probe_indices =
        Vec::with_capacity(probe.num_rows().min(JOIN_OUTPUT_CHUNK_ROWS));
    let mut output = Vec::new();
    let mut bloom_filtered_rows = 0_usize;
    let context = hash_join_materialize_context(
        probe,
        build,
        left_prefix,
        right_prefix,
        build_side,
        output_projection,
        metrics,
    )?;
    metrics.add_join_probe_rows(probe.num_rows());

    for probe_row in 0..probe.num_rows() {
        if key_array.is_null(probe_row) {
            push_unmatched_probe_match(
                &context,
                probe_row,
                join_type,
                &mut unmatched_probe_indices,
                &mut output,
            )?;
            continue;
        }
        let Some(build_ref) = i32_dense_rows.get(key_array.value(probe_row)) else {
            bloom_filtered_rows += 1;
            push_unmatched_probe_match(
                &context,
                probe_row,
                join_type,
                &mut unmatched_probe_indices,
                &mut output,
            )?;
            continue;
        };
        if let Some(matched_build) = matched_build.as_deref_mut() {
            matched_build.mark_i32_key(key_array.value(probe_row));
        }
        let probe_row = u32::try_from(probe_row).map_err(|_| {
            DodamError::UnsupportedSql(
                "hash join currently supports up to u32::MAX rows per side".to_string(),
            )
        })?;
        probe_indices.push(probe_row);
        build_refs.push(build_ref);
        if probe_indices.len() >= JOIN_OUTPUT_CHUNK_ROWS {
            output.push(materialize_hash_join_pairs(
                &context,
                &probe_indices,
                &build_refs,
            )?);
            probe_indices.clear();
            build_refs.clear();
        }
    }
    metrics.add_join_bloom_filtered_rows(bloom_filtered_rows);

    if !probe_indices.is_empty() {
        output.push(materialize_hash_join_pairs(
            &context,
            &probe_indices,
            &build_refs,
        )?);
    }
    if !unmatched_probe_indices.is_empty() {
        output.push(materialize_unmatched_probe_matches(
            &context,
            &unmatched_probe_indices,
        )?);
    }

    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn probe_i32_unique_hash_join_batches(
    probe: &RecordBatch,
    build: &HashJoinBuild,
    i32_unique_rows: &JoinKeyHashMap<i32, BuildRowRef>,
    probe_keys: &[String],
    left_prefix: &str,
    right_prefix: &str,
    build_side: JoinBuildSide,
    join_type: JoinType,
    mut matched_build: Option<&mut MatchedBuildTracker>,
    output_projection: &JoinOutputProjection,
    metrics: &ScanPlanMetricsCounter,
) -> Result<Vec<RecordBatch>> {
    let [probe_key] = probe_keys else {
        return Err(DodamError::UnsupportedSql(
            "Int32 unique hash join fast path expects exactly one probe key".to_string(),
        ));
    };
    let key_index = column_index(probe, probe_key)?;
    let key_array = probe.column(key_index);
    if key_array.data_type() != &DataType::Int32 {
        return Err(DodamError::UnsupportedSql(format!(
            "JOIN key types must match: probe side is {:?}, build side is {:?}",
            vec![key_array.data_type().clone()],
            build.key_data_types
        )));
    }
    let key_array = key_array
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("Int32 probe key array");
    let mut probe_indices = Vec::with_capacity(probe.num_rows().min(JOIN_OUTPUT_CHUNK_ROWS));
    let mut build_refs = Vec::with_capacity(probe.num_rows().min(JOIN_OUTPUT_CHUNK_ROWS));
    let mut output = Vec::new();
    let mut bloom_filtered_rows = 0_usize;
    let context = hash_join_materialize_context(
        probe,
        build,
        left_prefix,
        right_prefix,
        build_side,
        output_projection,
        metrics,
    )?;
    metrics.add_join_probe_rows(probe.num_rows());

    for probe_row in 0..probe.num_rows() {
        if key_array.is_null(probe_row) {
            push_unmatched_probe_if_outer(
                probe,
                build,
                probe_row,
                build_side,
                join_type,
                left_prefix,
                right_prefix,
                metrics,
                &mut output,
            )?;
            continue;
        }
        let Some(build_ref) = i32_unique_rows.get(&key_array.value(probe_row)).copied() else {
            bloom_filtered_rows += 1;
            push_unmatched_probe_if_outer(
                probe,
                build,
                probe_row,
                build_side,
                join_type,
                left_prefix,
                right_prefix,
                metrics,
                &mut output,
            )?;
            continue;
        };
        if let Some(matched_build) = matched_build.as_deref_mut() {
            matched_build.mark_ref(build_ref);
        }
        let probe_row = u32::try_from(probe_row).map_err(|_| {
            DodamError::UnsupportedSql(
                "hash join currently supports up to u32::MAX rows per side".to_string(),
            )
        })?;
        probe_indices.push(probe_row);
        build_refs.push(build_ref);
        if probe_indices.len() >= JOIN_OUTPUT_CHUNK_ROWS {
            output.push(materialize_hash_join_pairs(
                &context,
                &probe_indices,
                &build_refs,
            )?);
            probe_indices.clear();
            build_refs.clear();
        }
    }
    metrics.add_join_bloom_filtered_rows(bloom_filtered_rows);

    if !probe_indices.is_empty() {
        output.push(materialize_hash_join_pairs(
            &context,
            &probe_indices,
            &build_refs,
        )?);
    }

    Ok(output)
}

fn mark_build_matches(
    matched_build: &mut Option<&mut MatchedBuildTracker>,
    matches: &[BuildRowRef],
) {
    if let Some(matched_build) = matched_build.as_deref_mut() {
        matched_build.mark_refs(matches);
    }
}

#[allow(clippy::too_many_arguments)]
fn probe_i32_dense_hash_join_batches(
    probe: &RecordBatch,
    build: &HashJoinBuild,
    i32_rows: &DenseI32MultiBuildRows,
    probe_keys: &[String],
    left_prefix: &str,
    right_prefix: &str,
    build_side: JoinBuildSide,
    join_type: JoinType,
    mut matched_build: Option<&mut MatchedBuildTracker>,
    output_projection: &JoinOutputProjection,
    metrics: &ScanPlanMetricsCounter,
) -> Result<Vec<RecordBatch>> {
    let [probe_key] = probe_keys else {
        return Err(DodamError::UnsupportedSql(
            "Int32 hash join fast path expects exactly one probe key".to_string(),
        ));
    };
    let key_index = column_index(probe, probe_key)?;
    let key_array = probe.column(key_index);
    if key_array.data_type() != &DataType::Int32 {
        return Err(DodamError::UnsupportedSql(format!(
            "JOIN key types must match: probe side is {:?}, build side is {:?}",
            vec![key_array.data_type().clone()],
            build.key_data_types
        )));
    }
    let key_array = key_array
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("Int32 probe key array");
    let mut probe_indices = Vec::with_capacity(probe.num_rows().min(JOIN_OUTPUT_CHUNK_ROWS));
    let mut build_refs = Vec::with_capacity(probe.num_rows().min(JOIN_OUTPUT_CHUNK_ROWS));
    let mut semi_indices = Vec::with_capacity(probe.num_rows().min(JOIN_OUTPUT_CHUNK_ROWS));
    let mut output = Vec::new();
    let mut bloom_filtered_rows = 0_usize;
    let context = hash_join_materialize_context(
        probe,
        build,
        left_prefix,
        right_prefix,
        build_side,
        output_projection,
        metrics,
    )?;
    metrics.add_join_probe_rows(probe.num_rows());

    for probe_row in 0..probe.num_rows() {
        if key_array.is_null(probe_row) {
            push_unmatched_probe_if_outer(
                probe,
                build,
                probe_row,
                build_side,
                join_type,
                left_prefix,
                right_prefix,
                metrics,
                &mut output,
            )?;
            continue;
        }
        let Some(matches) = i32_rows.get(key_array.value(probe_row)) else {
            bloom_filtered_rows += 1;
            push_unmatched_probe_if_outer(
                probe,
                build,
                probe_row,
                build_side,
                join_type,
                left_prefix,
                right_prefix,
                metrics,
                &mut output,
            )?;
            continue;
        };
        mark_build_matches(&mut matched_build, matches);
        let probe_row = u32::try_from(probe_row).map_err(|_| {
            DodamError::UnsupportedSql(
                "hash join currently supports up to u32::MAX rows per side".to_string(),
            )
        })?;
        if join_type == JoinType::Semi {
            push_semi_join_match(&context, probe_row, &mut semi_indices, &mut output)?;
            continue;
        }
        push_hash_join_matches(
            &context,
            probe_row,
            matches,
            &mut probe_indices,
            &mut build_refs,
            &mut output,
        )?;
    }
    metrics.add_join_bloom_filtered_rows(bloom_filtered_rows);

    if !probe_indices.is_empty() {
        output.push(materialize_hash_join_pairs(
            &context,
            &probe_indices,
            &build_refs,
        )?);
    }
    if !semi_indices.is_empty() {
        output.push(materialize_semi_join_rows(&context, &semi_indices)?);
    }

    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn probe_i32_hash_join_batches(
    probe: &RecordBatch,
    build: &HashJoinBuild,
    i32_rows: &JoinKeyHashMap<i32, Vec<BuildRowRef>>,
    probe_keys: &[String],
    left_prefix: &str,
    right_prefix: &str,
    build_side: JoinBuildSide,
    join_type: JoinType,
    mut matched_build: Option<&mut MatchedBuildTracker>,
    output_projection: &JoinOutputProjection,
    metrics: &ScanPlanMetricsCounter,
) -> Result<Vec<RecordBatch>> {
    let [probe_key] = probe_keys else {
        return Err(DodamError::UnsupportedSql(
            "Int32 hash join fast path expects exactly one probe key".to_string(),
        ));
    };
    let key_index = column_index(probe, probe_key)?;
    let key_array = probe.column(key_index);
    if key_array.data_type() != &DataType::Int32 {
        return Err(DodamError::UnsupportedSql(format!(
            "JOIN key types must match: probe side is {:?}, build side is {:?}",
            vec![key_array.data_type().clone()],
            build.key_data_types
        )));
    }
    let key_array = key_array
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("Int32 probe key array");
    let mut probe_indices = Vec::new();
    let mut build_refs = Vec::new();
    let mut semi_indices = Vec::new();
    let mut output = Vec::new();
    let mut bloom_filtered_rows = 0_usize;
    let context = hash_join_materialize_context(
        probe,
        build,
        left_prefix,
        right_prefix,
        build_side,
        output_projection,
        metrics,
    )?;
    metrics.add_join_probe_rows(probe.num_rows());

    for probe_row in 0..probe.num_rows() {
        if key_array.is_null(probe_row) {
            push_unmatched_probe_if_outer(
                probe,
                build,
                probe_row,
                build_side,
                join_type,
                left_prefix,
                right_prefix,
                metrics,
                &mut output,
            )?;
            continue;
        }
        let Some(matches) = i32_rows.get(&key_array.value(probe_row)) else {
            bloom_filtered_rows += 1;
            push_unmatched_probe_if_outer(
                probe,
                build,
                probe_row,
                build_side,
                join_type,
                left_prefix,
                right_prefix,
                metrics,
                &mut output,
            )?;
            continue;
        };
        mark_build_matches(&mut matched_build, matches);
        let probe_row = u32::try_from(probe_row).map_err(|_| {
            DodamError::UnsupportedSql(
                "hash join currently supports up to u32::MAX rows per side".to_string(),
            )
        })?;
        if join_type == JoinType::Semi {
            push_semi_join_match(&context, probe_row, &mut semi_indices, &mut output)?;
            continue;
        }
        push_hash_join_matches(
            &context,
            probe_row,
            matches,
            &mut probe_indices,
            &mut build_refs,
            &mut output,
        )?;
    }
    metrics.add_join_bloom_filtered_rows(bloom_filtered_rows);

    if !probe_indices.is_empty() {
        output.push(materialize_hash_join_pairs(
            &context,
            &probe_indices,
            &build_refs,
        )?);
    }
    if !semi_indices.is_empty() {
        output.push(materialize_semi_join_rows(&context, &semi_indices)?);
    }

    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn probe_i64_dense_unique_hash_join_batches(
    probe: &RecordBatch,
    build: &HashJoinBuild,
    i64_dense_rows: &DenseI64BuildRows,
    probe_keys: &[String],
    left_prefix: &str,
    right_prefix: &str,
    build_side: JoinBuildSide,
    join_type: JoinType,
    mut matched_build: Option<&mut MatchedBuildTracker>,
    output_projection: &JoinOutputProjection,
    metrics: &ScanPlanMetricsCounter,
) -> Result<Vec<RecordBatch>> {
    let [probe_key] = probe_keys else {
        return Err(DodamError::UnsupportedSql(
            "Int64 dense hash join fast path expects exactly one probe key".to_string(),
        ));
    };
    let key_index = column_index(probe, probe_key)?;
    let key_array = probe.column(key_index);
    if key_array.data_type() != &DataType::Int64 {
        return Err(DodamError::UnsupportedSql(format!(
            "JOIN key types must match: probe side is {:?}, build side is {:?}",
            vec![key_array.data_type().clone()],
            build.key_data_types
        )));
    }
    let key_array = key_array
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("Int64 probe key array");
    let mut probe_indices = Vec::with_capacity(probe.num_rows().min(JOIN_OUTPUT_CHUNK_ROWS));
    let mut build_refs = Vec::with_capacity(probe.num_rows().min(JOIN_OUTPUT_CHUNK_ROWS));
    let mut output = Vec::new();
    let mut bloom_filtered_rows = 0_usize;
    let context = hash_join_materialize_context(
        probe,
        build,
        left_prefix,
        right_prefix,
        build_side,
        output_projection,
        metrics,
    )?;
    metrics.add_join_probe_rows(probe.num_rows());

    for probe_row in 0..probe.num_rows() {
        if key_array.is_null(probe_row) {
            push_unmatched_probe_if_outer(
                probe,
                build,
                probe_row,
                build_side,
                join_type,
                left_prefix,
                right_prefix,
                metrics,
                &mut output,
            )?;
            continue;
        }
        let Some(build_ref) = i64_dense_rows.get(key_array.value(probe_row)) else {
            bloom_filtered_rows += 1;
            push_unmatched_probe_if_outer(
                probe,
                build,
                probe_row,
                build_side,
                join_type,
                left_prefix,
                right_prefix,
                metrics,
                &mut output,
            )?;
            continue;
        };
        if let Some(matched_build) = matched_build.as_deref_mut() {
            matched_build.mark_ref(build_ref);
        }
        let probe_row = u32::try_from(probe_row).map_err(|_| {
            DodamError::UnsupportedSql(
                "hash join currently supports up to u32::MAX rows per side".to_string(),
            )
        })?;
        probe_indices.push(probe_row);
        build_refs.push(build_ref);
        if probe_indices.len() >= JOIN_OUTPUT_CHUNK_ROWS {
            output.push(materialize_hash_join_pairs(
                &context,
                &probe_indices,
                &build_refs,
            )?);
            probe_indices.clear();
            build_refs.clear();
        }
    }
    metrics.add_join_bloom_filtered_rows(bloom_filtered_rows);

    if !probe_indices.is_empty() {
        output.push(materialize_hash_join_pairs(
            &context,
            &probe_indices,
            &build_refs,
        )?);
    }

    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn probe_i64_unique_hash_join_batches(
    probe: &RecordBatch,
    build: &HashJoinBuild,
    i64_unique_rows: &JoinKeyHashMap<i64, BuildRowRef>,
    probe_keys: &[String],
    left_prefix: &str,
    right_prefix: &str,
    build_side: JoinBuildSide,
    join_type: JoinType,
    mut matched_build: Option<&mut MatchedBuildTracker>,
    output_projection: &JoinOutputProjection,
    metrics: &ScanPlanMetricsCounter,
) -> Result<Vec<RecordBatch>> {
    let [probe_key] = probe_keys else {
        return Err(DodamError::UnsupportedSql(
            "Int64 unique hash join fast path expects exactly one probe key".to_string(),
        ));
    };
    let key_index = column_index(probe, probe_key)?;
    let key_array = probe.column(key_index);
    if key_array.data_type() != &DataType::Int64 {
        return Err(DodamError::UnsupportedSql(format!(
            "JOIN key types must match: probe side is {:?}, build side is {:?}",
            vec![key_array.data_type().clone()],
            build.key_data_types
        )));
    }
    let key_array = key_array
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("Int64 probe key array");
    let mut probe_indices = Vec::with_capacity(probe.num_rows().min(JOIN_OUTPUT_CHUNK_ROWS));
    let mut build_refs = Vec::with_capacity(probe.num_rows().min(JOIN_OUTPUT_CHUNK_ROWS));
    let mut output = Vec::new();
    let mut bloom_filtered_rows = 0_usize;
    let context = hash_join_materialize_context(
        probe,
        build,
        left_prefix,
        right_prefix,
        build_side,
        output_projection,
        metrics,
    )?;
    metrics.add_join_probe_rows(probe.num_rows());

    for probe_row in 0..probe.num_rows() {
        if key_array.is_null(probe_row) {
            push_unmatched_probe_if_outer(
                probe,
                build,
                probe_row,
                build_side,
                join_type,
                left_prefix,
                right_prefix,
                metrics,
                &mut output,
            )?;
            continue;
        }
        let Some(build_ref) = i64_unique_rows.get(&key_array.value(probe_row)).copied() else {
            bloom_filtered_rows += 1;
            push_unmatched_probe_if_outer(
                probe,
                build,
                probe_row,
                build_side,
                join_type,
                left_prefix,
                right_prefix,
                metrics,
                &mut output,
            )?;
            continue;
        };
        if let Some(matched_build) = matched_build.as_deref_mut() {
            matched_build.mark_ref(build_ref);
        }
        let probe_row = u32::try_from(probe_row).map_err(|_| {
            DodamError::UnsupportedSql(
                "hash join currently supports up to u32::MAX rows per side".to_string(),
            )
        })?;
        probe_indices.push(probe_row);
        build_refs.push(build_ref);
        if probe_indices.len() >= JOIN_OUTPUT_CHUNK_ROWS {
            output.push(materialize_hash_join_pairs(
                &context,
                &probe_indices,
                &build_refs,
            )?);
            probe_indices.clear();
            build_refs.clear();
        }
    }
    metrics.add_join_bloom_filtered_rows(bloom_filtered_rows);

    if !probe_indices.is_empty() {
        output.push(materialize_hash_join_pairs(
            &context,
            &probe_indices,
            &build_refs,
        )?);
    }

    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn probe_i64_hash_join_batches(
    probe: &RecordBatch,
    build: &HashJoinBuild,
    i64_rows: &JoinKeyHashMap<i64, Vec<BuildRowRef>>,
    probe_keys: &[String],
    left_prefix: &str,
    right_prefix: &str,
    build_side: JoinBuildSide,
    join_type: JoinType,
    mut matched_build: Option<&mut MatchedBuildTracker>,
    output_projection: &JoinOutputProjection,
    metrics: &ScanPlanMetricsCounter,
) -> Result<Vec<RecordBatch>> {
    let [probe_key] = probe_keys else {
        return Err(DodamError::UnsupportedSql(
            "Int64 hash join fast path expects exactly one probe key".to_string(),
        ));
    };
    let key_index = column_index(probe, probe_key)?;
    let key_array = probe.column(key_index);
    if key_array.data_type() != &DataType::Int64 {
        return Err(DodamError::UnsupportedSql(format!(
            "JOIN key types must match: probe side is {:?}, build side is {:?}",
            vec![key_array.data_type().clone()],
            build.key_data_types
        )));
    }
    let key_array = key_array
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("Int64 probe key array");
    let mut probe_indices = Vec::new();
    let mut build_refs = Vec::new();
    let mut semi_indices = Vec::new();
    let mut output = Vec::new();
    let mut bloom_filtered_rows = 0_usize;
    let context = hash_join_materialize_context(
        probe,
        build,
        left_prefix,
        right_prefix,
        build_side,
        output_projection,
        metrics,
    )?;
    metrics.add_join_probe_rows(probe.num_rows());

    for probe_row in 0..probe.num_rows() {
        if key_array.is_null(probe_row) {
            push_unmatched_probe_if_outer(
                probe,
                build,
                probe_row,
                build_side,
                join_type,
                left_prefix,
                right_prefix,
                metrics,
                &mut output,
            )?;
            continue;
        }
        let Some(matches) = i64_rows.get(&key_array.value(probe_row)) else {
            bloom_filtered_rows += 1;
            push_unmatched_probe_if_outer(
                probe,
                build,
                probe_row,
                build_side,
                join_type,
                left_prefix,
                right_prefix,
                metrics,
                &mut output,
            )?;
            continue;
        };
        mark_build_matches(&mut matched_build, matches);
        let probe_row = u32::try_from(probe_row).map_err(|_| {
            DodamError::UnsupportedSql(
                "hash join currently supports up to u32::MAX rows per side".to_string(),
            )
        })?;
        if join_type == JoinType::Semi {
            push_semi_join_match(&context, probe_row, &mut semi_indices, &mut output)?;
            continue;
        }
        push_hash_join_matches(
            &context,
            probe_row,
            matches,
            &mut probe_indices,
            &mut build_refs,
            &mut output,
        )?;
    }
    metrics.add_join_bloom_filtered_rows(bloom_filtered_rows);

    if !probe_indices.is_empty() {
        output.push(materialize_hash_join_pairs(
            &context,
            &probe_indices,
            &build_refs,
        )?);
    }
    if !semi_indices.is_empty() {
        output.push(materialize_semi_join_rows(&context, &semi_indices)?);
    }

    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn probe_i32_pair_dense_unique_hash_join_batches(
    probe: &RecordBatch,
    build: &HashJoinBuild,
    i32_pair_rows: &DenseI32PairBuildRows,
    probe_keys: &[String],
    left_prefix: &str,
    right_prefix: &str,
    build_side: JoinBuildSide,
    join_type: JoinType,
    mut matched_build: Option<&mut MatchedBuildTracker>,
    output_projection: &JoinOutputProjection,
    metrics: &ScanPlanMetricsCounter,
) -> Result<Vec<RecordBatch>> {
    let [left_key, right_key] = probe_keys else {
        return Err(DodamError::UnsupportedSql(
            "Int32 pair dense unique hash join fast path expects exactly two probe keys"
                .to_string(),
        ));
    };
    let left_key_array = probe.column(column_index(probe, left_key)?);
    let right_key_array = probe.column(column_index(probe, right_key)?);
    let probe_key_types = vec![
        left_key_array.data_type().clone(),
        right_key_array.data_type().clone(),
    ];
    if probe_key_types != build.key_data_types {
        return Err(DodamError::UnsupportedSql(format!(
            "JOIN key types must match: probe side is {:?}, build side is {:?}",
            probe_key_types, build.key_data_types
        )));
    }
    let left_key_array = left_key_array
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("first Int32 probe key array");
    let right_key_array = right_key_array
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("second Int32 probe key array");
    let mut probe_indices = Vec::with_capacity(probe.num_rows().min(JOIN_OUTPUT_CHUNK_ROWS));
    let mut build_refs = Vec::with_capacity(probe.num_rows().min(JOIN_OUTPUT_CHUNK_ROWS));
    let mut semi_indices = Vec::with_capacity(probe.num_rows().min(JOIN_OUTPUT_CHUNK_ROWS));
    let mut output = Vec::new();
    let mut bloom_filtered_rows = 0_usize;
    let context = hash_join_materialize_context(
        probe,
        build,
        left_prefix,
        right_prefix,
        build_side,
        output_projection,
        metrics,
    )?;
    metrics.add_join_probe_rows(probe.num_rows());

    for probe_row in 0..probe.num_rows() {
        if left_key_array.is_null(probe_row) || right_key_array.is_null(probe_row) {
            push_unmatched_probe_if_outer(
                probe,
                build,
                probe_row,
                build_side,
                join_type,
                left_prefix,
                right_prefix,
                metrics,
                &mut output,
            )?;
            continue;
        }
        let Some(build_ref) = i32_pair_rows.get(
            left_key_array.value(probe_row),
            right_key_array.value(probe_row),
        ) else {
            bloom_filtered_rows += 1;
            push_unmatched_probe_if_outer(
                probe,
                build,
                probe_row,
                build_side,
                join_type,
                left_prefix,
                right_prefix,
                metrics,
                &mut output,
            )?;
            continue;
        };
        if let Some(matched_build) = matched_build.as_deref_mut() {
            matched_build.mark_ref(build_ref);
        }
        let probe_row = u32::try_from(probe_row).map_err(|_| {
            DodamError::UnsupportedSql(
                "hash join currently supports up to u32::MAX rows per side".to_string(),
            )
        })?;
        if join_type == JoinType::Semi {
            push_semi_join_match(&context, probe_row, &mut semi_indices, &mut output)?;
            continue;
        }
        probe_indices.push(probe_row);
        build_refs.push(build_ref);
        if probe_indices.len() >= JOIN_OUTPUT_CHUNK_ROWS {
            output.push(materialize_hash_join_pairs(
                &context,
                &probe_indices,
                &build_refs,
            )?);
            probe_indices.clear();
            build_refs.clear();
        }
    }
    metrics.add_join_bloom_filtered_rows(bloom_filtered_rows);

    if !probe_indices.is_empty() {
        output.push(materialize_hash_join_pairs(
            &context,
            &probe_indices,
            &build_refs,
        )?);
    }
    if !semi_indices.is_empty() {
        output.push(materialize_semi_join_rows(&context, &semi_indices)?);
    }

    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn probe_i32_pair_unique_hash_join_batches(
    probe: &RecordBatch,
    build: &HashJoinBuild,
    i32_pair_rows: &JoinKeyHashMap<(i32, i32), BuildRowRef>,
    probe_keys: &[String],
    left_prefix: &str,
    right_prefix: &str,
    build_side: JoinBuildSide,
    join_type: JoinType,
    mut matched_build: Option<&mut MatchedBuildTracker>,
    output_projection: &JoinOutputProjection,
    metrics: &ScanPlanMetricsCounter,
) -> Result<Vec<RecordBatch>> {
    let [left_key, right_key] = probe_keys else {
        return Err(DodamError::UnsupportedSql(
            "Int32 pair unique hash join fast path expects exactly two probe keys".to_string(),
        ));
    };
    let left_key_array = probe.column(column_index(probe, left_key)?);
    let right_key_array = probe.column(column_index(probe, right_key)?);
    let probe_key_types = vec![
        left_key_array.data_type().clone(),
        right_key_array.data_type().clone(),
    ];
    if probe_key_types != build.key_data_types {
        return Err(DodamError::UnsupportedSql(format!(
            "JOIN key types must match: probe side is {:?}, build side is {:?}",
            probe_key_types, build.key_data_types
        )));
    }
    let left_key_array = left_key_array
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("first Int32 probe key array");
    let right_key_array = right_key_array
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("second Int32 probe key array");
    let mut probe_indices = Vec::with_capacity(probe.num_rows().min(JOIN_OUTPUT_CHUNK_ROWS));
    let mut build_refs = Vec::with_capacity(probe.num_rows().min(JOIN_OUTPUT_CHUNK_ROWS));
    let mut semi_indices = Vec::with_capacity(probe.num_rows().min(JOIN_OUTPUT_CHUNK_ROWS));
    let mut output = Vec::new();
    let mut bloom_filtered_rows = 0_usize;
    let context = hash_join_materialize_context(
        probe,
        build,
        left_prefix,
        right_prefix,
        build_side,
        output_projection,
        metrics,
    )?;
    metrics.add_join_probe_rows(probe.num_rows());

    for probe_row in 0..probe.num_rows() {
        if left_key_array.is_null(probe_row) || right_key_array.is_null(probe_row) {
            push_unmatched_probe_if_outer(
                probe,
                build,
                probe_row,
                build_side,
                join_type,
                left_prefix,
                right_prefix,
                metrics,
                &mut output,
            )?;
            continue;
        }
        let key = (
            left_key_array.value(probe_row),
            right_key_array.value(probe_row),
        );
        let Some(build_ref) = i32_pair_rows.get(&key).copied() else {
            bloom_filtered_rows += 1;
            push_unmatched_probe_if_outer(
                probe,
                build,
                probe_row,
                build_side,
                join_type,
                left_prefix,
                right_prefix,
                metrics,
                &mut output,
            )?;
            continue;
        };
        if let Some(matched_build) = matched_build.as_deref_mut() {
            matched_build.mark_ref(build_ref);
        }
        let probe_row = u32::try_from(probe_row).map_err(|_| {
            DodamError::UnsupportedSql(
                "hash join currently supports up to u32::MAX rows per side".to_string(),
            )
        })?;
        if join_type == JoinType::Semi {
            push_semi_join_match(&context, probe_row, &mut semi_indices, &mut output)?;
            continue;
        }
        probe_indices.push(probe_row);
        build_refs.push(build_ref);
        if probe_indices.len() >= JOIN_OUTPUT_CHUNK_ROWS {
            output.push(materialize_hash_join_pairs(
                &context,
                &probe_indices,
                &build_refs,
            )?);
            probe_indices.clear();
            build_refs.clear();
        }
    }
    metrics.add_join_bloom_filtered_rows(bloom_filtered_rows);

    if !probe_indices.is_empty() {
        output.push(materialize_hash_join_pairs(
            &context,
            &probe_indices,
            &build_refs,
        )?);
    }
    if !semi_indices.is_empty() {
        output.push(materialize_semi_join_rows(&context, &semi_indices)?);
    }

    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn probe_i32_pair_hash_join_batches(
    probe: &RecordBatch,
    build: &HashJoinBuild,
    i32_pair_rows: &JoinKeyHashMap<(i32, i32), Vec<BuildRowRef>>,
    probe_keys: &[String],
    left_prefix: &str,
    right_prefix: &str,
    build_side: JoinBuildSide,
    join_type: JoinType,
    mut matched_build: Option<&mut MatchedBuildTracker>,
    output_projection: &JoinOutputProjection,
    metrics: &ScanPlanMetricsCounter,
) -> Result<Vec<RecordBatch>> {
    let [left_key, right_key] = probe_keys else {
        return Err(DodamError::UnsupportedSql(
            "Int32 pair hash join fast path expects exactly two probe keys".to_string(),
        ));
    };
    let left_key_array = probe.column(column_index(probe, left_key)?);
    let right_key_array = probe.column(column_index(probe, right_key)?);
    let probe_key_types = vec![
        left_key_array.data_type().clone(),
        right_key_array.data_type().clone(),
    ];
    if probe_key_types != build.key_data_types {
        return Err(DodamError::UnsupportedSql(format!(
            "JOIN key types must match: probe side is {:?}, build side is {:?}",
            probe_key_types, build.key_data_types
        )));
    }
    let left_key_array = left_key_array
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("first Int32 probe key array");
    let right_key_array = right_key_array
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("second Int32 probe key array");
    let mut probe_indices = Vec::new();
    let mut build_refs = Vec::new();
    let mut semi_indices = Vec::new();
    let mut output = Vec::new();
    let mut bloom_filtered_rows = 0_usize;
    let context = hash_join_materialize_context(
        probe,
        build,
        left_prefix,
        right_prefix,
        build_side,
        output_projection,
        metrics,
    )?;
    metrics.add_join_probe_rows(probe.num_rows());

    for probe_row in 0..probe.num_rows() {
        if left_key_array.is_null(probe_row) || right_key_array.is_null(probe_row) {
            push_unmatched_probe_if_outer(
                probe,
                build,
                probe_row,
                build_side,
                join_type,
                left_prefix,
                right_prefix,
                metrics,
                &mut output,
            )?;
            continue;
        }
        let key = (
            left_key_array.value(probe_row),
            right_key_array.value(probe_row),
        );
        let Some(matches) = i32_pair_rows.get(&key) else {
            bloom_filtered_rows += 1;
            push_unmatched_probe_if_outer(
                probe,
                build,
                probe_row,
                build_side,
                join_type,
                left_prefix,
                right_prefix,
                metrics,
                &mut output,
            )?;
            continue;
        };
        mark_build_matches(&mut matched_build, matches);
        let probe_row = u32::try_from(probe_row).map_err(|_| {
            DodamError::UnsupportedSql(
                "hash join currently supports up to u32::MAX rows per side".to_string(),
            )
        })?;
        if join_type == JoinType::Semi {
            push_semi_join_match(&context, probe_row, &mut semi_indices, &mut output)?;
            continue;
        }
        push_hash_join_matches(
            &context,
            probe_row,
            matches,
            &mut probe_indices,
            &mut build_refs,
            &mut output,
        )?;
    }
    metrics.add_join_bloom_filtered_rows(bloom_filtered_rows);

    if !probe_indices.is_empty() {
        output.push(materialize_hash_join_pairs(
            &context,
            &probe_indices,
            &build_refs,
        )?);
    }
    if !semi_indices.is_empty() {
        output.push(materialize_semi_join_rows(&context, &semi_indices)?);
    }

    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn probe_string_unique_hash_join_batches(
    probe: &RecordBatch,
    build: &HashJoinBuild,
    string_rows: &JoinKeyHashMap<String, BuildRowRef>,
    probe_keys: &[String],
    left_prefix: &str,
    right_prefix: &str,
    build_side: JoinBuildSide,
    join_type: JoinType,
    mut matched_build: Option<&mut MatchedBuildTracker>,
    output_projection: &JoinOutputProjection,
    metrics: &ScanPlanMetricsCounter,
) -> Result<Vec<RecordBatch>> {
    let [probe_key] = probe_keys else {
        return Err(DodamError::UnsupportedSql(
            "String unique hash join fast path expects exactly one probe key".to_string(),
        ));
    };
    let key_index = column_index(probe, probe_key)?;
    let key_array = probe.column(key_index);
    if key_array.data_type() != &DataType::Utf8 {
        return Err(DodamError::UnsupportedSql(format!(
            "JOIN key types must match: probe side is {:?}, build side is {:?}",
            vec![key_array.data_type().clone()],
            build.key_data_types
        )));
    }
    let key_array = key_array
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("Utf8 probe key array");
    let mut probe_indices = Vec::with_capacity(probe.num_rows().min(JOIN_OUTPUT_CHUNK_ROWS));
    let mut build_refs = Vec::with_capacity(probe.num_rows().min(JOIN_OUTPUT_CHUNK_ROWS));
    let mut semi_indices = Vec::with_capacity(probe.num_rows().min(JOIN_OUTPUT_CHUNK_ROWS));
    let mut output = Vec::new();
    let mut bloom_filtered_rows = 0_usize;
    let context = hash_join_materialize_context(
        probe,
        build,
        left_prefix,
        right_prefix,
        build_side,
        output_projection,
        metrics,
    )?;
    metrics.add_join_probe_rows(probe.num_rows());

    for probe_row in 0..probe.num_rows() {
        if key_array.is_null(probe_row) {
            push_unmatched_probe_if_outer(
                probe,
                build,
                probe_row,
                build_side,
                join_type,
                left_prefix,
                right_prefix,
                metrics,
                &mut output,
            )?;
            continue;
        }
        let Some(build_ref) = string_rows.get(key_array.value(probe_row)).copied() else {
            bloom_filtered_rows += 1;
            push_unmatched_probe_if_outer(
                probe,
                build,
                probe_row,
                build_side,
                join_type,
                left_prefix,
                right_prefix,
                metrics,
                &mut output,
            )?;
            continue;
        };
        if let Some(matched_build) = matched_build.as_deref_mut() {
            matched_build.mark_ref(build_ref);
        }
        let probe_row = u32::try_from(probe_row).map_err(|_| {
            DodamError::UnsupportedSql(
                "hash join currently supports up to u32::MAX rows per side".to_string(),
            )
        })?;
        if join_type == JoinType::Semi {
            push_semi_join_match(&context, probe_row, &mut semi_indices, &mut output)?;
            continue;
        }
        probe_indices.push(probe_row);
        build_refs.push(build_ref);
        if probe_indices.len() >= JOIN_OUTPUT_CHUNK_ROWS {
            output.push(materialize_hash_join_pairs(
                &context,
                &probe_indices,
                &build_refs,
            )?);
            probe_indices.clear();
            build_refs.clear();
        }
    }
    metrics.add_join_bloom_filtered_rows(bloom_filtered_rows);

    if !probe_indices.is_empty() {
        output.push(materialize_hash_join_pairs(
            &context,
            &probe_indices,
            &build_refs,
        )?);
    }
    if !semi_indices.is_empty() {
        output.push(materialize_semi_join_rows(&context, &semi_indices)?);
    }

    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn probe_string_hash_join_batches(
    probe: &RecordBatch,
    build: &HashJoinBuild,
    string_rows: &JoinKeyHashMap<String, Vec<BuildRowRef>>,
    probe_keys: &[String],
    left_prefix: &str,
    right_prefix: &str,
    build_side: JoinBuildSide,
    join_type: JoinType,
    mut matched_build: Option<&mut MatchedBuildTracker>,
    output_projection: &JoinOutputProjection,
    metrics: &ScanPlanMetricsCounter,
) -> Result<Vec<RecordBatch>> {
    let [probe_key] = probe_keys else {
        return Err(DodamError::UnsupportedSql(
            "String hash join fast path expects exactly one probe key".to_string(),
        ));
    };
    let key_array = probe.column(column_index(probe, probe_key)?);
    if key_array.data_type() != &DataType::Utf8 {
        return Err(DodamError::UnsupportedSql(format!(
            "JOIN key types must match: probe side is {:?}, build side is {:?}",
            vec![key_array.data_type().clone()],
            build.key_data_types
        )));
    }
    let key_array = key_array
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("Utf8 probe key array");
    let mut probe_indices = Vec::new();
    let mut build_refs = Vec::new();
    let mut semi_indices = Vec::new();
    let mut output = Vec::new();
    let mut bloom_filtered_rows = 0_usize;
    let context = hash_join_materialize_context(
        probe,
        build,
        left_prefix,
        right_prefix,
        build_side,
        output_projection,
        metrics,
    )?;
    metrics.add_join_probe_rows(probe.num_rows());

    for probe_row in 0..probe.num_rows() {
        if key_array.is_null(probe_row) {
            push_unmatched_probe_if_outer(
                probe,
                build,
                probe_row,
                build_side,
                join_type,
                left_prefix,
                right_prefix,
                metrics,
                &mut output,
            )?;
            continue;
        }
        let Some(matches) = string_rows.get(key_array.value(probe_row)) else {
            bloom_filtered_rows += 1;
            push_unmatched_probe_if_outer(
                probe,
                build,
                probe_row,
                build_side,
                join_type,
                left_prefix,
                right_prefix,
                metrics,
                &mut output,
            )?;
            continue;
        };
        mark_build_matches(&mut matched_build, matches);
        let probe_row = u32::try_from(probe_row).map_err(|_| {
            DodamError::UnsupportedSql(
                "hash join currently supports up to u32::MAX rows per side".to_string(),
            )
        })?;
        if join_type == JoinType::Semi {
            push_semi_join_match(&context, probe_row, &mut semi_indices, &mut output)?;
            continue;
        }
        push_hash_join_matches(
            &context,
            probe_row,
            matches,
            &mut probe_indices,
            &mut build_refs,
            &mut output,
        )?;
    }
    metrics.add_join_bloom_filtered_rows(bloom_filtered_rows);

    if !probe_indices.is_empty() {
        output.push(materialize_hash_join_pairs(
            &context,
            &probe_indices,
            &build_refs,
        )?);
    }
    if !semi_indices.is_empty() {
        output.push(materialize_semi_join_rows(&context, &semi_indices)?);
    }

    Ok(output)
}

fn mark_probe_match(matched_probe_keys: &mut Option<&mut HashSet<OwnedRow>>, key: &OwnedRow) {
    if let Some(matched_probe_keys) = matched_probe_keys.as_deref_mut() {
        matched_probe_keys.insert(key.clone());
    }
}

fn push_hash_join_matches(
    context: &HashJoinMaterializeContext<'_>,
    probe_row: u32,
    matches: &[BuildRowRef],
    probe_indices: &mut Vec<u32>,
    build_refs: &mut Vec<BuildRowRef>,
    output: &mut Vec<RecordBatch>,
) -> Result<()> {
    for build_ref in matches {
        probe_indices.push(probe_row);
        build_refs.push(*build_ref);
        if probe_indices.len() >= JOIN_OUTPUT_CHUNK_ROWS {
            output.push(materialize_hash_join_pairs(
                context,
                probe_indices,
                build_refs,
            )?);
            probe_indices.clear();
            build_refs.clear();
        }
    }
    Ok(())
}

fn push_semi_join_match(
    context: &HashJoinMaterializeContext<'_>,
    probe_row: u32,
    probe_indices: &mut Vec<u32>,
    output: &mut Vec<RecordBatch>,
) -> Result<()> {
    if context.build_side != JoinBuildSide::Right {
        return Err(DodamError::UnsupportedSql(
            "SEMI JOIN currently expects the right side as hash build input".to_string(),
        ));
    }
    probe_indices.push(probe_row);
    if probe_indices.len() >= JOIN_OUTPUT_CHUNK_ROWS {
        output.push(materialize_semi_join_rows(context, probe_indices)?);
        probe_indices.clear();
    }
    Ok(())
}

fn push_unmatched_probe_match(
    context: &HashJoinMaterializeContext<'_>,
    probe_row: usize,
    join_type: JoinType,
    probe_indices: &mut Vec<u32>,
    output: &mut Vec<RecordBatch>,
) -> Result<()> {
    let should_emit = matches!(
        (join_type, context.build_side),
        (JoinType::Left, JoinBuildSide::Right)
            | (JoinType::Right, JoinBuildSide::Left)
            | (JoinType::Full, _)
    );
    if !should_emit {
        return Ok(());
    }
    let probe_row = u32::try_from(probe_row).map_err(|_| {
        DodamError::UnsupportedSql(
            "outer hash join currently supports up to u32::MAX rows per batch".to_string(),
        )
    })?;
    probe_indices.push(probe_row);
    if probe_indices.len() >= JOIN_OUTPUT_CHUNK_ROWS {
        output.push(materialize_unmatched_probe_matches(context, probe_indices)?);
        probe_indices.clear();
    }
    Ok(())
}

fn materialize_unmatched_probe_matches(
    context: &HashJoinMaterializeContext<'_>,
    probe_indices: &[u32],
) -> Result<RecordBatch> {
    let started = Instant::now();
    let (probe_columns, build_columns) = match context.build_side {
        JoinBuildSide::Left => (
            context.output_projection.right_columns.as_ref(),
            context.output_projection.left_columns.as_ref(),
        ),
        JoinBuildSide::Right => (
            context.output_projection.left_columns.as_ref(),
            context.output_projection.right_columns.as_ref(),
        ),
    };
    let probe_taken =
        take_record_batch_by_indices_projected(context.probe, probe_indices, probe_columns)?;
    let build_template = project_batch_columns(&context.build.batches[0], build_columns)?;
    let null_build =
        null_record_batch_for_schema(build_template.schema().as_ref(), probe_indices.len())?;
    let (left_taken, right_taken) = match context.build_side {
        JoinBuildSide::Left => (null_build, probe_taken),
        JoinBuildSide::Right => (probe_taken, null_build),
    };
    context.metrics.add_join_output_rows(left_taken.num_rows());
    let batch = join_output_batch_with_schema(
        &left_taken,
        &right_taken,
        context.join_output_schema.clone(),
    )?;
    context.metrics.add_join_materialize_time(started.elapsed());
    Ok(batch)
}

fn materialize_semi_join_rows(
    context: &HashJoinMaterializeContext<'_>,
    probe_indices: &[u32],
) -> Result<RecordBatch> {
    let started = Instant::now();
    let left_taken = take_record_batch_by_indices_projected(
        context.probe,
        probe_indices,
        context.output_projection.left_columns.as_ref(),
    )?;
    context.metrics.add_join_output_rows(left_taken.num_rows());
    let batch =
        single_side_output_batch_with_schema(&left_taken, context.semi_output_schema.clone())?;
    context.metrics.add_join_materialize_time(started.elapsed());
    Ok(batch)
}

fn materialize_hash_join_pairs(
    context: &HashJoinMaterializeContext<'_>,
    probe_indices: &[u32],
    build_refs: &[BuildRowRef],
) -> Result<RecordBatch> {
    let started = Instant::now();
    let (probe_columns, build_columns) = match context.build_side {
        JoinBuildSide::Left => (
            context.output_projection.right_columns.as_ref(),
            context.output_projection.left_columns.as_ref(),
        ),
        JoinBuildSide::Right => (
            context.output_projection.left_columns.as_ref(),
            context.output_projection.right_columns.as_ref(),
        ),
    };
    let probe_taken =
        take_record_batch_by_indices_projected(context.probe, probe_indices, probe_columns)?;
    let build_taken = take_build_row_refs_projected(context.build, build_refs, build_columns)?;
    let (left_taken, right_taken) = match context.build_side {
        JoinBuildSide::Left => (build_taken, probe_taken),
        JoinBuildSide::Right => (probe_taken, build_taken),
    };
    context.metrics.add_join_output_rows(left_taken.num_rows());
    let batch = join_output_batch_with_schema(
        &left_taken,
        &right_taken,
        context.join_output_schema.clone(),
    )?;
    context.metrics.add_join_materialize_time(started.elapsed());
    Ok(batch)
}

fn single_side_output_batch_with_schema(
    batch: &RecordBatch,
    schema: Arc<Schema>,
) -> Result<RecordBatch> {
    Ok(RecordBatch::try_new(schema, batch.columns().to_vec())?)
}

fn take_record_batch_by_indices(batch: &RecordBatch, indices: &[u32]) -> Result<RecordBatch> {
    if let Some((start, len)) = contiguous_index_range(indices) {
        return Ok(batch.slice(start, len));
    }
    Ok(take_record_batch(
        batch,
        &UInt32Array::from(indices.to_vec()),
    )?)
}

fn take_record_batch_by_indices_projected(
    batch: &RecordBatch,
    indices: &[u32],
    columns: Option<&Vec<String>>,
) -> Result<RecordBatch> {
    let projected = project_batch_columns(batch, columns)?;
    take_record_batch_by_indices(&projected, indices)
}

fn project_batch_columns(
    batch: &RecordBatch,
    columns: Option<&Vec<String>>,
) -> Result<RecordBatch> {
    let Some(columns) = columns else {
        return Ok(batch.clone());
    };
    if columns.len() == batch.num_columns()
        && columns
            .iter()
            .zip(batch.schema().fields())
            .all(|(column, field)| column == field.name())
    {
        return Ok(batch.clone());
    }
    let indices = columns
        .iter()
        .map(|column| column_index(batch, column))
        .collect::<Result<Vec<_>>>()?;
    Ok(batch.project(&indices)?)
}

fn contiguous_index_range(indices: &[u32]) -> Option<(usize, usize)> {
    let (&first, rest) = indices.split_first()?;
    let mut expected = first;
    for &index in rest {
        expected = expected.checked_add(1)?;
        if index != expected {
            return None;
        }
    }
    Some((first as usize, indices.len()))
}

fn take_build_row_refs(build: &HashJoinBuild, refs: &[BuildRowRef]) -> Result<RecordBatch> {
    take_build_row_refs_projected(build, refs, None)
}

fn take_build_row_refs_projected(
    build: &HashJoinBuild,
    refs: &[BuildRowRef],
    columns: Option<&Vec<String>>,
) -> Result<RecordBatch> {
    if let Some((batch_index, start, len)) = contiguous_build_ref_range(refs) {
        let batch = build.batches.get(batch_index).ok_or_else(|| {
            DodamError::UnsupportedSql("hash join build row reference is out of range".to_string())
        })?;
        let projected = project_batch_columns(batch, columns)?;
        return Ok(projected.slice(start, len));
    }

    if let Some(batch) = try_take_build_row_refs_direct(build, refs, columns)? {
        return Ok(batch);
    }

    if should_take_build_refs_via_global_indices(refs) {
        return take_build_row_refs_via_global_indices(build, refs, columns);
    }

    let schema = build.batches[0].schema();
    let mut chunks = Vec::new();
    let mut offset = 0_usize;
    while offset < refs.len() {
        let batch_index = refs[offset].batch;
        let mut rows = Vec::new();
        while offset < refs.len() && refs[offset].batch == batch_index {
            rows.push(refs[offset].row);
            offset += 1;
        }
        let batch = build.batches.get(batch_index).ok_or_else(|| {
            DodamError::UnsupportedSql("hash join build row reference is out of range".to_string())
        })?;
        chunks.push(take_record_batch_by_indices_projected(
            batch, &rows, columns,
        )?);
    }
    if chunks.len() == 1 {
        return Ok(chunks.remove(0));
    }
    let schema = chunks.first().map(RecordBatch::schema).unwrap_or(schema);
    Ok(concat_batches(&schema, chunks.iter())?)
}

fn try_take_build_row_refs_direct(
    build: &HashJoinBuild,
    refs: &[BuildRowRef],
    columns: Option<&Vec<String>>,
) -> Result<Option<RecordBatch>> {
    if !should_take_build_refs_via_global_indices(refs) {
        return Ok(None);
    }
    let schema = build.batches[0].schema();
    let column_indices = match columns {
        Some(columns) => columns
            .iter()
            .map(|column| column_index(&build.batches[0], column))
            .collect::<Result<Vec<_>>>()?,
        None => (0..schema.fields().len()).collect(),
    };
    let mut arrays = Vec::with_capacity(column_indices.len());
    let mut fields = Vec::with_capacity(column_indices.len());
    for column_index in column_indices {
        let field = schema.field(column_index).clone();
        let Some(array) = try_gather_build_column(build, refs, column_index, field.data_type())?
        else {
            return Ok(None);
        };
        fields.push(field);
        arrays.push(array);
    }
    Ok(Some(RecordBatch::try_new(
        Arc::new(Schema::new(fields)),
        arrays,
    )?))
}

fn try_gather_build_column(
    build: &HashJoinBuild,
    refs: &[BuildRowRef],
    column_index: usize,
    data_type: &DataType,
) -> Result<Option<ArrayRef>> {
    match data_type {
        DataType::Int32 => {
            let arrays = build_column_arrays::<Int32Array>(build, column_index)?;
            let mut builder = Int32Builder::with_capacity(refs.len());
            for row_ref in refs {
                let array = arrays.get(row_ref.batch).ok_or_else(|| {
                    DodamError::UnsupportedSql(
                        "hash join build row reference is out of range".to_string(),
                    )
                })?;
                let row = row_ref.row as usize;
                if array.is_null(row) {
                    builder.append_null();
                } else {
                    builder.append_value(array.value(row));
                }
            }
            Ok(Some(Arc::new(builder.finish())))
        }
        DataType::Int64 => {
            let arrays = build_column_arrays::<Int64Array>(build, column_index)?;
            let mut builder = Int64Builder::with_capacity(refs.len());
            for row_ref in refs {
                let array = arrays.get(row_ref.batch).ok_or_else(|| {
                    DodamError::UnsupportedSql(
                        "hash join build row reference is out of range".to_string(),
                    )
                })?;
                let row = row_ref.row as usize;
                if array.is_null(row) {
                    builder.append_null();
                } else {
                    builder.append_value(array.value(row));
                }
            }
            Ok(Some(Arc::new(builder.finish())))
        }
        DataType::UInt32 => {
            let arrays = build_column_arrays::<UInt32Array>(build, column_index)?;
            let mut builder = UInt32Builder::with_capacity(refs.len());
            for row_ref in refs {
                let array = arrays.get(row_ref.batch).ok_or_else(|| {
                    DodamError::UnsupportedSql(
                        "hash join build row reference is out of range".to_string(),
                    )
                })?;
                let row = row_ref.row as usize;
                if array.is_null(row) {
                    builder.append_null();
                } else {
                    builder.append_value(array.value(row));
                }
            }
            Ok(Some(Arc::new(builder.finish())))
        }
        DataType::UInt64 => {
            let arrays = build_column_arrays::<UInt64Array>(build, column_index)?;
            let mut builder = UInt64Builder::with_capacity(refs.len());
            for row_ref in refs {
                let array = arrays.get(row_ref.batch).ok_or_else(|| {
                    DodamError::UnsupportedSql(
                        "hash join build row reference is out of range".to_string(),
                    )
                })?;
                let row = row_ref.row as usize;
                if array.is_null(row) {
                    builder.append_null();
                } else {
                    builder.append_value(array.value(row));
                }
            }
            Ok(Some(Arc::new(builder.finish())))
        }
        DataType::Float64 => {
            let arrays = build_column_arrays::<Float64Array>(build, column_index)?;
            let mut builder = Float64Builder::with_capacity(refs.len());
            for row_ref in refs {
                let array = arrays.get(row_ref.batch).ok_or_else(|| {
                    DodamError::UnsupportedSql(
                        "hash join build row reference is out of range".to_string(),
                    )
                })?;
                let row = row_ref.row as usize;
                if array.is_null(row) {
                    builder.append_null();
                } else {
                    builder.append_value(array.value(row));
                }
            }
            Ok(Some(Arc::new(builder.finish())))
        }
        DataType::Utf8 => {
            let arrays = build_column_arrays::<StringArray>(build, column_index)?;
            let mut builder =
                StringBuilder::with_capacity(refs.len(), refs.len().saturating_mul(8));
            for row_ref in refs {
                let array = arrays.get(row_ref.batch).ok_or_else(|| {
                    DodamError::UnsupportedSql(
                        "hash join build row reference is out of range".to_string(),
                    )
                })?;
                let row = row_ref.row as usize;
                if array.is_null(row) {
                    builder.append_null();
                } else {
                    builder.append_value(array.value(row));
                }
            }
            Ok(Some(Arc::new(builder.finish())))
        }
        DataType::Dictionary(key_type, value_type)
            if matches!(
                (&**key_type, &**value_type),
                (DataType::Int32, DataType::Utf8)
            ) =>
        {
            let arrays = build_column_arrays::<DictionaryArray<Int32Type>>(build, column_index)?;
            if arrays.len() != 1 {
                return Ok(None);
            }
            let array = arrays[0];
            let mut builder = Int32Builder::with_capacity(refs.len());
            for row_ref in refs {
                if row_ref.batch != 0 {
                    return Ok(None);
                }
                let row = row_ref.row as usize;
                if array.is_null(row) {
                    builder.append_null();
                } else {
                    builder.append_value(array.keys().value(row));
                }
            }
            Ok(Some(Arc::new(DictionaryArray::new(
                builder.finish(),
                array.values().clone(),
            ))))
        }
        DataType::Boolean => {
            let arrays = build_column_arrays::<BooleanArray>(build, column_index)?;
            let mut builder = BooleanBuilder::with_capacity(refs.len());
            for row_ref in refs {
                let array = arrays.get(row_ref.batch).ok_or_else(|| {
                    DodamError::UnsupportedSql(
                        "hash join build row reference is out of range".to_string(),
                    )
                })?;
                let row = row_ref.row as usize;
                if array.is_null(row) {
                    builder.append_null();
                } else {
                    builder.append_value(array.value(row));
                }
            }
            Ok(Some(Arc::new(builder.finish())))
        }
        _ => Ok(None),
    }
}

fn build_column_arrays<T: 'static>(build: &HashJoinBuild, column_index: usize) -> Result<Vec<&T>> {
    build
        .batches
        .iter()
        .map(|batch| {
            batch
                .column(column_index)
                .as_any()
                .downcast_ref::<T>()
                .ok_or_else(|| {
                    DodamError::UnsupportedSql(
                        "hash join build column type changed across batches".to_string(),
                    )
                })
        })
        .collect()
}

fn should_take_build_refs_via_global_indices(refs: &[BuildRowRef]) -> bool {
    let mut run_count = 0_usize;
    let mut previous_batch = None;
    for row_ref in refs {
        if previous_batch != Some(row_ref.batch) {
            run_count += 1;
            previous_batch = Some(row_ref.batch);
        }
        if run_count > 8 {
            return true;
        }
    }
    false
}

fn take_build_row_refs_via_global_indices(
    build: &HashJoinBuild,
    refs: &[BuildRowRef],
    columns: Option<&Vec<String>>,
) -> Result<RecordBatch> {
    let mut batch_indices = Vec::new();
    for row_ref in refs {
        if !batch_indices.contains(&row_ref.batch) {
            batch_indices.push(row_ref.batch);
        }
    }
    batch_indices.sort_unstable();

    let mut projected_batches = Vec::with_capacity(batch_indices.len());
    let mut offsets = HashMap::with_capacity(batch_indices.len());
    let mut offset = 0_u32;
    for batch_index in batch_indices {
        let batch = build.batches.get(batch_index).ok_or_else(|| {
            DodamError::UnsupportedSql("hash join build row reference is out of range".to_string())
        })?;
        let projected = project_batch_columns(batch, columns)?;
        offsets.insert(batch_index, offset);
        offset = offset
            .checked_add(u32::try_from(projected.num_rows()).map_err(|_| {
                DodamError::UnsupportedSql(
                    "hash join build side currently supports up to u32::MAX rows".to_string(),
                )
            })?)
            .ok_or_else(|| {
                DodamError::UnsupportedSql(
                    "hash join build side currently supports up to u32::MAX rows".to_string(),
                )
            })?;
        projected_batches.push(projected);
    }

    let schema = projected_batches
        .first()
        .map(RecordBatch::schema)
        .unwrap_or_else(|| build.batches[0].schema());
    let concatenated = if projected_batches.len() == 1 {
        projected_batches.remove(0)
    } else {
        concat_batches(&schema, projected_batches.iter())?
    };
    let indices = refs
        .iter()
        .map(|row_ref| {
            offsets
                .get(&row_ref.batch)
                .copied()
                .and_then(|offset| offset.checked_add(row_ref.row))
                .ok_or_else(|| {
                    DodamError::UnsupportedSql(
                        "hash join build row reference is out of range".to_string(),
                    )
                })
        })
        .collect::<Result<Vec<_>>>()?;
    take_record_batch_by_indices(&concatenated, &indices)
}

fn contiguous_build_ref_range(refs: &[BuildRowRef]) -> Option<(usize, usize, usize)> {
    let (&first, rest) = refs.split_first()?;
    let mut expected = first.row;
    for row_ref in rest {
        if row_ref.batch != first.batch {
            return None;
        }
        expected = expected.checked_add(1)?;
        if row_ref.row != expected {
            return None;
        }
    }
    Some((first.batch, first.row as usize, refs.len()))
}

#[allow(clippy::too_many_arguments)]
fn push_unmatched_probe_if_outer(
    probe: &RecordBatch,
    build: &HashJoinBuild,
    probe_row: usize,
    build_side: JoinBuildSide,
    join_type: JoinType,
    left_prefix: &str,
    right_prefix: &str,
    metrics: &ScanPlanMetricsCounter,
    output: &mut Vec<RecordBatch>,
) -> Result<()> {
    let should_emit = matches!(
        (join_type, build_side),
        (JoinType::Left, JoinBuildSide::Right)
            | (JoinType::Right, JoinBuildSide::Left)
            | (JoinType::Full, _)
    );
    if !should_emit {
        return Ok(());
    }
    let probe_row = u32::try_from(probe_row).map_err(|_| {
        DodamError::UnsupportedSql(
            "outer hash join currently supports up to u32::MAX rows per batch".to_string(),
        )
    })?;
    let probe_taken = probe.slice(probe_row as usize, 1);
    let null_build = null_record_batch_like(&build.batches[0], 1)?;
    let (left_taken, right_taken) = match build_side {
        JoinBuildSide::Left => (null_build, probe_taken),
        JoinBuildSide::Right => (probe_taken, null_build),
    };
    metrics.add_join_output_rows(left_taken.num_rows());
    output.push(join_output_batch(
        &left_taken,
        &right_taken,
        left_prefix,
        right_prefix,
    )?);
    Ok(())
}

fn materialize_unmatched_probe_by_matched_keys(
    probe: &RecordBatch,
    probe_keys: &[String],
    matched_probe_keys: &HashSet<OwnedRow>,
    context: &UnmatchedProbeMaterializeContext<'_>,
) -> Result<Vec<RecordBatch>> {
    let probe_key_arrays = key_arrays(probe, probe_keys)?;
    let probe_converter = RowConverter::new(
        probe_key_arrays
            .iter()
            .map(|array| SortField::new(array.data_type().clone()))
            .collect(),
    )?;
    let probe_rows = probe_converter.convert_columns(&probe_key_arrays)?;
    let mut output = Vec::new();
    let mut unmatched = Vec::new();

    for (row, key) in probe_rows.iter().enumerate() {
        let is_unmatched = probe_key_arrays.iter().any(|array| array.is_null(row))
            || !matched_probe_keys.contains(&key.owned());
        if !is_unmatched {
            continue;
        }
        unmatched.push(u32::try_from(row).map_err(|_| {
            DodamError::UnsupportedSql(
                "outer hash join currently supports up to u32::MAX rows per batch".to_string(),
            )
        })?);
        if unmatched.len() >= JOIN_OUTPUT_CHUNK_ROWS {
            output.push(materialize_unmatched_probe_rows(
                probe, &unmatched, context,
            )?);
            unmatched.clear();
        }
    }

    if !unmatched.is_empty() {
        output.push(materialize_unmatched_probe_rows(
            probe, &unmatched, context,
        )?);
    }
    Ok(output)
}

fn materialize_unmatched_probe_rows(
    probe: &RecordBatch,
    probe_indices: &[u32],
    context: &UnmatchedProbeMaterializeContext<'_>,
) -> Result<RecordBatch> {
    let probe_taken = take_record_batch_by_indices(probe, probe_indices)?;
    let null_build = null_record_batch_for_schema(context.build_schema, probe_taken.num_rows())?;
    let (left_taken, right_taken) = match context.build_side {
        JoinBuildSide::Left => (null_build, probe_taken),
        JoinBuildSide::Right => (probe_taken, null_build),
    };
    context.metrics.add_join_output_rows(left_taken.num_rows());
    join_output_batch(
        &left_taken,
        &right_taken,
        context.left_prefix,
        context.right_prefix,
    )
}

#[allow(clippy::too_many_arguments)]
fn emit_unmatched_build_if_needed(
    build: &HashJoinBuild,
    matched_build: Option<&MatchedBuildTracker>,
    build_side: JoinBuildSide,
    join_type: JoinType,
    left_prefix: &str,
    right_prefix: &str,
    probe_template: Option<&RecordBatch>,
    metrics: &ScanPlanMetricsCounter,
    emitted: &mut bool,
) -> Result<Vec<RecordBatch>> {
    if !should_emit_unmatched_build(join_type, build_side) || *emitted {
        return Ok(Vec::new());
    }
    *emitted = true;
    let Some(probe_template) = probe_template else {
        return Ok(Vec::new());
    };
    let default_tracker;
    let matched_build = match matched_build {
        Some(matched_build) => matched_build,
        None => {
            default_tracker = MatchedBuildTracker::empty_set();
            &default_tracker
        }
    };

    if matched_build.all_dense_i32_matched() {
        return Ok(Vec::new());
    }

    let mut output = Vec::new();
    let mut unmatched = Vec::new();
    if let (Some(dense_rows), MatchedBuildTracker::DenseI32 { matched, .. }) =
        (&build.i32_dense_unique_rows, matched_build)
    {
        for build_ref in dense_rows.unmatched_refs(matched) {
            unmatched.push(build_ref);
            if unmatched.len() >= JOIN_OUTPUT_CHUNK_ROWS {
                output.push(materialize_unmatched_build_rows(
                    build,
                    &unmatched,
                    probe_template,
                    build_side,
                    left_prefix,
                    right_prefix,
                    metrics,
                )?);
                unmatched.clear();
            }
        }
    } else {
        for build_ref in &build.all_rows {
            if !matched_build.is_matched(build_ref) {
                unmatched.push(*build_ref);
                if unmatched.len() >= JOIN_OUTPUT_CHUNK_ROWS {
                    output.push(materialize_unmatched_build_rows(
                        build,
                        &unmatched,
                        probe_template,
                        build_side,
                        left_prefix,
                        right_prefix,
                        metrics,
                    )?);
                    unmatched.clear();
                }
            }
        }
    }
    if !unmatched.is_empty() {
        output.push(materialize_unmatched_build_rows(
            build,
            &unmatched,
            probe_template,
            build_side,
            left_prefix,
            right_prefix,
            metrics,
        )?);
    }
    Ok(output)
}

fn should_track_matched_build(join_type: JoinType, build_side: JoinBuildSide) -> bool {
    should_emit_unmatched_build(join_type, build_side)
}

fn should_emit_unmatched_build(join_type: JoinType, build_side: JoinBuildSide) -> bool {
    matches!(
        (join_type, build_side),
        (JoinType::Full, _)
            | (JoinType::Left, JoinBuildSide::Left)
            | (JoinType::Right, JoinBuildSide::Right)
    )
}

fn materialize_unmatched_build_rows(
    build: &HashJoinBuild,
    build_refs: &[BuildRowRef],
    probe_template: &RecordBatch,
    build_side: JoinBuildSide,
    left_prefix: &str,
    right_prefix: &str,
    metrics: &ScanPlanMetricsCounter,
) -> Result<RecordBatch> {
    let build_taken = take_build_row_refs(build, build_refs)?;
    let null_probe = null_record_batch_like(probe_template, build_taken.num_rows())?;
    let (left_taken, right_taken) = match build_side {
        JoinBuildSide::Left => (build_taken, null_probe),
        JoinBuildSide::Right => (null_probe, build_taken),
    };
    metrics.add_join_output_rows(left_taken.num_rows());
    join_output_batch(&left_taken, &right_taken, left_prefix, right_prefix)
}

fn null_record_batch_like(batch: &RecordBatch, rows: usize) -> Result<RecordBatch> {
    null_record_batch_for_schema(&batch.schema(), rows)
}

fn null_record_batch_for_schema(schema: &Schema, rows: usize) -> Result<RecordBatch> {
    let nullable_schema = Arc::new(Schema::new(
        schema
            .fields()
            .iter()
            .map(|field| Field::new(field.name(), field.data_type().clone(), true))
            .collect::<Vec<_>>(),
    ));
    let columns = schema
        .fields()
        .iter()
        .map(|field| new_null_array(field.data_type(), rows))
        .collect::<Vec<_>>();
    Ok(RecordBatch::try_new(nullable_schema, columns)?)
}

fn materialize_join_pairs(
    context: &JoinMaterializeContext<'_>,
    probe_indices: &[u32],
    build_indices: &[u32],
) -> Result<RecordBatch> {
    let (probe_columns, build_columns) = match context.build_side {
        JoinBuildSide::Left => (
            context.output_projection.right_columns.as_ref(),
            context.output_projection.left_columns.as_ref(),
        ),
        JoinBuildSide::Right => (
            context.output_projection.left_columns.as_ref(),
            context.output_projection.right_columns.as_ref(),
        ),
    };
    let probe_taken =
        take_record_batch_by_indices_projected(context.probe, probe_indices, probe_columns)?;
    let build_taken =
        take_record_batch_by_indices_projected(context.build, build_indices, build_columns)?;
    let (left_taken, right_taken) = match context.build_side {
        JoinBuildSide::Left => (build_taken, probe_taken),
        JoinBuildSide::Right => (probe_taken, build_taken),
    };
    context.metrics.add_join_output_rows(left_taken.num_rows());
    join_output_batch(
        &left_taken,
        &right_taken,
        context.left_prefix,
        context.right_prefix,
    )
}

#[allow(clippy::too_many_arguments)]
fn block_nested_loop_join_batches(
    left: &RecordBatch,
    right: &RecordBatch,
    left_keys: &[String],
    right_keys: &[String],
    left_prefix: &str,
    right_prefix: &str,
    output_projection: &JoinOutputProjection,
    metrics: &ScanPlanMetricsCounter,
) -> Result<Vec<RecordBatch>> {
    let left_key_arrays = key_arrays(left, left_keys)?;
    let right_key_arrays = key_arrays(right, right_keys)?;
    let left_key_types = left_key_arrays
        .iter()
        .map(|array| array.data_type().clone())
        .collect::<Vec<_>>();
    let right_key_types = right_key_arrays
        .iter()
        .map(|array| array.data_type().clone())
        .collect::<Vec<_>>();
    if left_key_types != right_key_types {
        return Err(DodamError::UnsupportedSql(format!(
            "JOIN key types must match: left side is {:?}, right side is {:?}",
            left_key_types, right_key_types
        )));
    }

    let left_converter = RowConverter::new(
        left_key_arrays
            .iter()
            .map(|array| SortField::new(array.data_type().clone()))
            .collect(),
    )?;
    let right_converter = RowConverter::new(
        right_key_arrays
            .iter()
            .map(|array| SortField::new(array.data_type().clone()))
            .collect(),
    )?;
    let left_rows = left_converter.convert_columns(&left_key_arrays)?;
    let right_rows = right_converter.convert_columns(&right_key_arrays)?;
    let context = JoinMaterializeContext {
        probe: left,
        build: right,
        left_prefix,
        right_prefix,
        build_side: JoinBuildSide::Right,
        output_projection,
        metrics,
    };
    let mut left_indices = Vec::new();
    let mut right_indices = Vec::new();
    let mut output = Vec::new();

    metrics.add_join_probe_rows(left.num_rows());
    metrics.add_join_build_rows(right.num_rows());
    metrics.observe_join_build_bytes(record_batch_memory_size(right));

    for left_row in 0..left.num_rows() {
        if left_key_arrays.iter().any(|array| array.is_null(left_row)) {
            continue;
        }
        let left_key_row = left_rows.row(left_row).owned();
        let left_row = u32::try_from(left_row).map_err(|_| {
            DodamError::UnsupportedSql(
                "block nested loop join currently supports up to u32::MAX rows per batch"
                    .to_string(),
            )
        })?;
        for right_row in 0..right.num_rows() {
            if right_key_arrays
                .iter()
                .any(|array| array.is_null(right_row))
            {
                continue;
            }
            if right_rows.row(right_row).owned() != left_key_row {
                continue;
            }
            let right_row = u32::try_from(right_row).map_err(|_| {
                DodamError::UnsupportedSql(
                    "block nested loop join currently supports up to u32::MAX rows per batch"
                        .to_string(),
                )
            })?;
            left_indices.push(left_row);
            right_indices.push(right_row);
            if left_indices.len() >= JOIN_OUTPUT_CHUNK_ROWS {
                output.push(materialize_join_pairs(
                    &context,
                    &left_indices,
                    &right_indices,
                )?);
                left_indices.clear();
                right_indices.clear();
            }
        }
    }

    if !left_indices.is_empty() {
        output.push(materialize_join_pairs(
            &context,
            &left_indices,
            &right_indices,
        )?);
    }
    Ok(output)
}

fn sort_merge_join_batches(
    left_batches: &[RecordBatch],
    right_batches: &[RecordBatch],
    left_key: &str,
    right_key: &str,
    left_prefix: &str,
    right_prefix: &str,
    metrics: &ScanPlanMetricsCounter,
) -> Result<RecordBatch> {
    let left_schema = left_batches[0].schema();
    let right_schema = right_batches[0].schema();
    let left = concat_batches(&left_schema, left_batches.iter())?;
    let right = concat_batches(&right_schema, right_batches.iter())?;
    let left = sort_single_batch_by_column(left, left_key)?;
    let right = sort_single_batch_by_column(right, right_key)?;
    let left_key_index = column_index(&left, left_key)?;
    let right_key_index = column_index(&right, right_key)?;
    let left_key_array = left.column(left_key_index);
    let right_key_array = right.column(right_key_index);
    if left_key_array.data_type() != right_key_array.data_type() {
        return Err(DodamError::UnsupportedSql(format!(
            "JOIN key types must match: {} is {:?}, {} is {:?}",
            left_key,
            left_key_array.data_type(),
            right_key,
            right_key_array.data_type()
        )));
    }

    let sort_field = SortField::new(left_key_array.data_type().clone());
    let left_converter = RowConverter::new(vec![sort_field.clone()])?;
    let right_converter = RowConverter::new(vec![sort_field])?;
    let left_rows = left_converter.convert_columns(std::slice::from_ref(left_key_array))?;
    let right_rows = right_converter.convert_columns(std::slice::from_ref(right_key_array))?;
    let mut left_indices = Vec::new();
    let mut right_indices = Vec::new();
    let mut left_row = 0_usize;
    let mut right_row = 0_usize;

    metrics.add_join_build_rows(right.num_rows());
    metrics.observe_join_build_bytes(record_batch_memory_size(&right));
    metrics.add_join_probe_rows(left.num_rows());

    while left_row < left.num_rows() && right_row < right.num_rows() {
        if left_key_array.is_null(left_row) {
            left_row += 1;
            continue;
        }
        if right_key_array.is_null(right_row) {
            right_row += 1;
            continue;
        }

        let left_key_row = left_rows.row(left_row).owned();
        let right_key_row = right_rows.row(right_row).owned();
        if left_key_row < right_key_row {
            left_row += 1;
            continue;
        }
        if left_key_row > right_key_row {
            right_row += 1;
            continue;
        }

        let left_start = left_row;
        while left_row < left.num_rows()
            && !left_key_array.is_null(left_row)
            && left_rows.row(left_row).owned() == left_key_row
        {
            left_row += 1;
        }
        let right_start = right_row;
        while right_row < right.num_rows()
            && !right_key_array.is_null(right_row)
            && right_rows.row(right_row).owned() == right_key_row
        {
            right_row += 1;
        }

        for left_index in left_start..left_row {
            let left_index = u32::try_from(left_index).map_err(|_| {
                DodamError::UnsupportedSql(
                    "sort-merge join currently supports up to u32::MAX rows per side".to_string(),
                )
            })?;
            for right_index in right_start..right_row {
                let right_index = u32::try_from(right_index).map_err(|_| {
                    DodamError::UnsupportedSql(
                        "sort-merge join currently supports up to u32::MAX rows per side"
                            .to_string(),
                    )
                })?;
                left_indices.push(left_index);
                right_indices.push(right_index);
            }
        }
    }

    if left_indices.is_empty() {
        return empty_join_batch(&left, &right, left_prefix, right_prefix);
    }

    let left_taken = take_record_batch(&left, &UInt32Array::from(left_indices))?;
    let right_taken = take_record_batch(&right, &UInt32Array::from(right_indices))?;
    metrics.add_join_output_rows(left_taken.num_rows());
    join_output_batch(&left_taken, &right_taken, left_prefix, right_prefix)
}

fn sort_single_batch_by_column(batch: RecordBatch, column: &str) -> Result<RecordBatch> {
    sort_batches(
        &[batch],
        &SortKey::from(SortExpr {
            column: column.to_string(),
            descending: false,
            nulls_first: false,
        }),
        None,
    )
}

fn empty_join_batch(
    left: &RecordBatch,
    right: &RecordBatch,
    left_prefix: &str,
    right_prefix: &str,
) -> Result<RecordBatch> {
    let fields = qualified_fields(left, left_prefix)
        .into_iter()
        .chain(qualified_fields(right, right_prefix))
        .collect::<Vec<_>>();
    let columns = left
        .columns()
        .iter()
        .map(|column| column.slice(0, 0))
        .chain(right.columns().iter().map(|column| column.slice(0, 0)))
        .collect::<Vec<_>>();
    Ok(RecordBatch::try_new(
        Arc::new(Schema::new(fields)),
        columns,
    )?)
}

fn join_output_batch(
    left: &RecordBatch,
    right: &RecordBatch,
    left_prefix: &str,
    right_prefix: &str,
) -> Result<RecordBatch> {
    join_output_batch_with_schema(
        left,
        right,
        Arc::new(Schema::new(
            qualified_fields(left, left_prefix)
                .into_iter()
                .chain(qualified_fields(right, right_prefix))
                .collect::<Vec<_>>(),
        )),
    )
}

fn join_output_batch_with_schema(
    left: &RecordBatch,
    right: &RecordBatch,
    schema: Arc<Schema>,
) -> Result<RecordBatch> {
    let columns = left
        .columns()
        .iter()
        .cloned()
        .chain(right.columns().iter().cloned())
        .collect::<Vec<_>>();
    Ok(RecordBatch::try_new(schema, columns)?)
}

fn qualified_fields(batch: &RecordBatch, prefix: &str) -> Vec<Field> {
    batch
        .schema()
        .fields()
        .iter()
        .map(|field| {
            Field::new(
                format!("{prefix}.{}", field.name()),
                field.data_type().clone(),
                true,
            )
        })
        .collect()
}

pub struct LimitExec {
    input: Box<dyn PhysicalPlan>,
    limit: usize,
}

impl LimitExec {
    pub fn new(input: Box<dyn PhysicalPlan>, limit: usize) -> Self {
        Self { input, limit }
    }
}

impl PhysicalPlan for LimitExec {
    fn execute(self: Box<Self>) -> Result<SendableBatchStream> {
        let input = self.input.execute()?;
        let (input, metrics) = input.into_parts();
        Ok(SendableBatchStream::new(
            Box::new(LimitStream {
                input,
                remaining: self.limit,
                limit_nanos: 0,
                metrics: metrics.clone(),
            }),
            metrics,
        ))
    }
}

struct LimitStream {
    input: Box<dyn Iterator<Item = Result<RecordBatch>> + Send>,
    remaining: usize,
    limit_nanos: u64,
    metrics: Arc<ScanPlanMetricsCounter>,
}

impl Iterator for LimitStream {
    type Item = Result<RecordBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            self.flush_limit_time();
            return None;
        }

        for batch in &mut self.input {
            match batch {
                Ok(batch) if batch.num_rows() == 0 => continue,
                Ok(batch) => {
                    let start = Instant::now();
                    let rows = self.remaining.min(batch.num_rows());
                    self.remaining -= rows;
                    let batch = batch.slice(0, rows);
                    self.limit_nanos = self.limit_nanos.saturating_add(elapsed_nanos(start));
                    return Some(Ok(batch));
                }
                Err(error) => return Some(Err(error)),
            }
        }

        self.flush_limit_time();
        None
    }
}

impl LimitStream {
    fn flush_limit_time(&mut self) {
        if self.limit_nanos == 0 {
            return;
        }

        self.metrics
            .add_limit_time(Duration::from_nanos(self.limit_nanos));
        self.limit_nanos = 0;
    }
}

impl Drop for LimitStream {
    fn drop(&mut self) {
        self.flush_limit_time();
    }
}

pub fn collect_metrics(
    mut stream: SendableBatchStream,
    fragments: usize,
    output_projection: &Projection,
) -> Result<ScanMetrics> {
    let mut metrics = ScanMetrics {
        fragments,
        ..ScanMetrics::default()
    };

    for batch in stream.by_ref() {
        let batch = batch?;
        if batch.num_rows() == 0 {
            continue;
        }

        metrics.batches += 1;
        metrics.rows += batch.num_rows();
        metrics.columns = output_column_count(output_projection, batch.num_columns());
    }

    let scan_plan_metrics = stream.into_scan_plan_metrics();
    metrics.row_groups_total = scan_plan_metrics.row_groups_total;
    metrics.row_groups_scanned = scan_plan_metrics.row_groups_scanned;
    metrics.row_groups_pruned = scan_plan_metrics.row_groups_pruned;
    metrics.compressed_bytes_total = scan_plan_metrics.compressed_bytes_total;
    metrics.compressed_bytes_scanned = scan_plan_metrics.compressed_bytes_scanned;
    metrics.compressed_bytes_pruned = scan_plan_metrics.compressed_bytes_pruned;
    metrics.metadata_nanos = scan_plan_metrics.metadata_nanos;
    metrics.planning_nanos = scan_plan_metrics.planning_nanos;
    metrics.decode_nanos = scan_plan_metrics.decode_nanos;
    metrics.filter_nanos = scan_plan_metrics.filter_nanos;
    metrics.projection_nanos = scan_plan_metrics.projection_nanos;
    metrics.limit_nanos = scan_plan_metrics.limit_nanos;
    metrics.join_build_rows = scan_plan_metrics.join_build_rows;
    metrics.join_probe_rows = scan_plan_metrics.join_probe_rows;
    metrics.join_output_rows = scan_plan_metrics.join_output_rows;
    metrics.join_spill_files = scan_plan_metrics.join_spill_files;
    metrics.join_spill_bytes = scan_plan_metrics.join_spill_bytes;
    metrics.join_repartitions = scan_plan_metrics.join_repartitions;
    metrics.join_heavy_hitters = scan_plan_metrics.join_heavy_hitters;
    metrics.join_bloom_filtered_rows = scan_plan_metrics.join_bloom_filtered_rows;
    metrics.join_nested_loop_fallbacks = scan_plan_metrics.join_nested_loop_fallbacks;
    metrics.join_peak_build_bytes = scan_plan_metrics.join_peak_build_bytes;

    Ok(metrics)
}

pub fn scan_projection(projection: &Projection, filter: Option<&FilterExpr>) -> Projection {
    let Some(filter) = filter else {
        return projection.clone();
    };

    let Projection::Columns(columns) = projection else {
        return Projection::All;
    };

    let mut columns = columns.clone();
    for filter_column in filter.referenced_columns() {
        if !columns.iter().any(|column| column == &filter_column) {
            columns.push(filter_column);
        }
    }

    Projection::Columns(columns)
}

pub fn filter_batch(batch: RecordBatch, filter: &FilterExpr) -> Result<RecordBatch> {
    let mask = evaluate_filter_mask(&batch, filter)?;
    Ok(filter_record_batch(&batch, &mask)?)
}

pub fn evaluate_filter_mask(batch: &RecordBatch, filter: &FilterExpr) -> Result<BooleanArray> {
    evaluate_filter(batch, filter.expr())
}

fn evaluate_filter(batch: &RecordBatch, expr: &Expr) -> Result<BooleanArray> {
    match expr {
        Expr::Boolean(value) => Ok(BooleanArray::from(vec![*value; batch.num_rows()])),
        Expr::Comparison(comparison) => evaluate_comparison(batch, comparison),
        Expr::ColumnComparison { left, op, right } => {
            evaluate_column_comparison(batch, left, *op, right)
        }
        Expr::InList {
            column,
            values,
            negated,
            has_null,
        } => evaluate_in_list(batch, column, values, *negated, *has_null),
        Expr::Like {
            column,
            pattern,
            negated,
            escape,
        } => evaluate_like(batch, column, pattern, *negated, *escape),
        Expr::IsNull { column, negated } => evaluate_is_null(batch, column, *negated),
        Expr::Not(expr) => {
            let mask = evaluate_filter(batch, expr)?;
            Ok(not_mask(&mask))
        }
        Expr::And(left, right) => {
            let left = evaluate_filter(batch, left)?;
            let right = evaluate_filter(batch, right)?;
            Ok(and_masks(&left, &right))
        }
        Expr::Or(left, right) => {
            let left = evaluate_filter(batch, left)?;
            let right = evaluate_filter(batch, right)?;
            Ok(or_masks(&left, &right))
        }
    }
}

fn evaluate_comparison(batch: &RecordBatch, comparison: &ComparisonExpr) -> Result<BooleanArray> {
    if matches!(comparison.value, crate::execution::LiteralValue::Null) {
        return Ok(BooleanArray::from(vec![None; batch.num_rows()]));
    }
    let column_index = column_index(batch, &comparison.column)?;
    let column = batch.column(column_index);
    let mask = match column.data_type() {
        DataType::Int32 => {
            let value = comparison.value.as_i32(&comparison.column)?;
            let scalar = Int32Array::from(vec![value]);
            compare(column, &Scalar::new(scalar), comparison.op)?
        }
        DataType::Int64 => {
            let value = comparison.value.as_i64(&comparison.column)?;
            let scalar = Int64Array::from(vec![value]);
            compare(column, &Scalar::new(scalar), comparison.op)?
        }
        DataType::UInt64 => {
            let value = comparison.value.as_u64(&comparison.column)?;
            let scalar = UInt64Array::from(vec![value]);
            compare(column, &Scalar::new(scalar), comparison.op)?
        }
        DataType::Float64 => {
            let value = comparison.value.as_f64(&comparison.column)?;
            let scalar = Float64Array::from(vec![value]);
            compare(column, &Scalar::new(scalar), comparison.op)?
        }
        DataType::Boolean => {
            let value = comparison.value.as_bool(&comparison.column)?;
            let scalar = BooleanArray::from(vec![value]);
            compare(column, &Scalar::new(scalar), comparison.op)?
        }
        DataType::Date32 => {
            let value = date_literal_as_i32(&comparison.value, &comparison.column)?;
            let scalar = Date32Array::from(vec![value]);
            compare(column, &Scalar::new(scalar), comparison.op)?
        }
        DataType::Date64 => {
            let value = date_literal_as_i64(&comparison.value, &comparison.column)?;
            let scalar = Date64Array::from(vec![value]);
            compare(column, &Scalar::new(scalar), comparison.op)?
        }
        DataType::Decimal128(precision, scale) => {
            let value = decimal_literal_as_i128(&comparison.value, *scale, &comparison.column)?;
            let scalar = Decimal128Array::from(vec![value])
                .with_precision_and_scale(*precision, *scale)
                .map_err(DodamError::Arrow)?;
            compare(column, &Scalar::new(scalar), comparison.op)?
        }
        DataType::Timestamp(unit, timezone) => match unit {
            TimeUnit::Second => {
                let value = timestamp_literal_as_i64(&comparison.value, *unit, &comparison.column)?;
                let scalar =
                    TimestampSecondArray::from(vec![value]).with_timezone_opt(timezone.clone());
                compare(column, &Scalar::new(scalar), comparison.op)?
            }
            TimeUnit::Millisecond => {
                let value = timestamp_literal_as_i64(&comparison.value, *unit, &comparison.column)?;
                let scalar = TimestampMillisecondArray::from(vec![value])
                    .with_timezone_opt(timezone.clone());
                compare(column, &Scalar::new(scalar), comparison.op)?
            }
            TimeUnit::Microsecond => {
                let value = timestamp_literal_as_i64(&comparison.value, *unit, &comparison.column)?;
                let scalar = TimestampMicrosecondArray::from(vec![value])
                    .with_timezone_opt(timezone.clone());
                compare(column, &Scalar::new(scalar), comparison.op)?
            }
            TimeUnit::Nanosecond => {
                let value = timestamp_literal_as_i64(&comparison.value, *unit, &comparison.column)?;
                let scalar =
                    TimestampNanosecondArray::from(vec![value]).with_timezone_opt(timezone.clone());
                compare(column, &Scalar::new(scalar), comparison.op)?
            }
        },
        DataType::Utf8 => {
            let value = comparison.value.as_str();
            let scalar = StringArray::from(vec![value.as_str()]);
            compare(column, &Scalar::new(scalar), comparison.op)?
        }
        data_type => {
            return Err(DodamError::UnsupportedFilterType {
                column: comparison.column.clone(),
                data_type: data_type.clone(),
            });
        }
    };

    Ok(mask)
}

fn evaluate_column_comparison(
    batch: &RecordBatch,
    left: &str,
    op: ComparisonOp,
    right: &str,
) -> Result<BooleanArray> {
    let left_index = column_index(batch, left)?;
    let right_index = column_index(batch, right)?;
    let left_column = batch.column(left_index);
    let right_column = batch.column(right_index);
    if matches!(left_column.data_type(), DataType::Decimal128(_, _))
        && matches!(right_column.data_type(), DataType::Decimal128(_, _))
    {
        return compare_decimal128_columns(left_column, right_column, op)
            .map_err(|_| DodamError::InvalidFilter(format!("{left} {op:?} {right}")));
    }
    if left_column.data_type() != right_column.data_type() {
        return Err(DodamError::InvalidFilter(format!("{left} {op:?} {right}")));
    }
    Ok(compare_columns(left_column, right_column, op)?)
}

fn decimal_literal_as_i128(
    value: &crate::execution::LiteralValue,
    scale: i8,
    column: &str,
) -> Result<i128> {
    let scale = usize::try_from(scale)
        .map_err(|_| DodamError::InvalidFilter(format!("{column}={value}")))?;
    match value {
        crate::execution::LiteralValue::Int64(value) => Ok(i128::from(*value)
            .checked_mul(decimal_scale_factor(scale)?)
            .ok_or_else(|| DodamError::InvalidFilter(format!("{column}={value}")))?),
        crate::execution::LiteralValue::Float64(value) => {
            decimal_string_as_i128(&value.to_string(), scale, column)
        }
        crate::execution::LiteralValue::Utf8(value) => decimal_string_as_i128(value, scale, column),
        crate::execution::LiteralValue::Null | crate::execution::LiteralValue::Boolean(_) => {
            Err(DodamError::InvalidFilter(format!("{column}={value}")))
        }
    }
}

fn decimal_string_as_i128(value: &str, scale: usize, column: &str) -> Result<i128> {
    let value = value.trim();
    if value.is_empty() {
        return Err(DodamError::InvalidFilter(format!("{column}={value}")));
    }
    let negative = value.starts_with('-');
    let unsigned = value.strip_prefix(['-', '+']).unwrap_or(value);
    let (whole, fractional) = unsigned.split_once('.').unwrap_or((unsigned, ""));
    if whole.is_empty() || !whole.chars().all(|ch| ch.is_ascii_digit()) {
        return Err(DodamError::InvalidFilter(format!("{column}={value}")));
    }
    if !fractional.chars().all(|ch| ch.is_ascii_digit()) {
        return Err(DodamError::InvalidFilter(format!("{column}={value}")));
    }
    let fractional = &fractional[..fractional.len().min(scale)];
    let mut result = whole
        .parse::<i128>()
        .map_err(|_| DodamError::InvalidFilter(format!("{column}={value}")))?
        .checked_mul(decimal_scale_factor(scale)?)
        .ok_or_else(|| DodamError::InvalidFilter(format!("{column}={value}")))?;
    let mut fractional_value = fractional.parse::<i128>().unwrap_or(0);
    for _ in fractional.len()..scale {
        fractional_value = fractional_value
            .checked_mul(10)
            .ok_or_else(|| DodamError::InvalidFilter(format!("{column}={value}")))?;
    }
    result = result
        .checked_add(fractional_value)
        .ok_or_else(|| DodamError::InvalidFilter(format!("{column}={value}")))?;
    if negative {
        result = -result;
    }
    Ok(result)
}

fn decimal_scale_factor(scale: usize) -> Result<i128> {
    let mut factor = 1_i128;
    for _ in 0..scale {
        factor = factor
            .checked_mul(10)
            .ok_or_else(|| DodamError::InvalidFilter("decimal scale is too large".to_string()))?;
    }
    Ok(factor)
}

fn timestamp_literal_as_i64(
    value: &crate::execution::LiteralValue,
    unit: TimeUnit,
    column: &str,
) -> Result<i64> {
    let millis = match value {
        crate::execution::LiteralValue::Int64(value) => *value,
        crate::execution::LiteralValue::Utf8(value) => parse_timestamp_millis(value)
            .ok_or_else(|| DodamError::InvalidFilter(format!("{column}={value}")))?,
        crate::execution::LiteralValue::Null
        | crate::execution::LiteralValue::Boolean(_)
        | crate::execution::LiteralValue::Float64(_) => {
            return Err(DodamError::InvalidFilter(format!("{column}={value}")));
        }
    };
    match unit {
        TimeUnit::Second => Ok(millis / 1_000),
        TimeUnit::Millisecond => Ok(millis),
        TimeUnit::Microsecond => millis
            .checked_mul(1_000)
            .ok_or_else(|| DodamError::InvalidFilter(format!("{column}={value}"))),
        TimeUnit::Nanosecond => millis
            .checked_mul(1_000_000)
            .ok_or_else(|| DodamError::InvalidFilter(format!("{column}={value}"))),
    }
}

fn date_literal_as_i32(value: &crate::execution::LiteralValue, column: &str) -> Result<i32> {
    match value {
        crate::execution::LiteralValue::Int64(value) => i32::try_from(*value)
            .map_err(|_| DodamError::InvalidFilter(format!("{column}={value}"))),
        crate::execution::LiteralValue::Utf8(value) => parse_date_days(value)
            .and_then(|value| i32::try_from(value).ok())
            .ok_or_else(|| DodamError::InvalidFilter(format!("{column}={value}"))),
        crate::execution::LiteralValue::Null
        | crate::execution::LiteralValue::Boolean(_)
        | crate::execution::LiteralValue::Float64(_) => {
            Err(DodamError::InvalidFilter(format!("{column}={value}")))
        }
    }
}

fn date_literal_as_i64(value: &crate::execution::LiteralValue, column: &str) -> Result<i64> {
    match value {
        crate::execution::LiteralValue::Int64(value) => Ok(*value),
        crate::execution::LiteralValue::Utf8(value) => parse_date_days(value)
            .and_then(|days| days.checked_mul(86_400_000))
            .ok_or_else(|| DodamError::InvalidFilter(format!("{column}={value}"))),
        crate::execution::LiteralValue::Null
        | crate::execution::LiteralValue::Boolean(_)
        | crate::execution::LiteralValue::Float64(_) => {
            Err(DodamError::InvalidFilter(format!("{column}={value}")))
        }
    }
}

fn parse_date_days(value: &str) -> Option<i64> {
    let date = value
        .trim()
        .split_once([' ', 'T'])
        .map_or(value.trim(), |(date, _)| date);
    let mut date_parts = date.split('-');
    let year = date_parts.next()?.parse::<i32>().ok()?;
    let month = date_parts.next()?.parse::<u32>().ok()?;
    let day = date_parts.next()?.parse::<u32>().ok()?;
    if date_parts.next().is_some() || !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }
    days_from_civil(year, month, day)
}

fn parse_timestamp_millis(value: &str) -> Option<i64> {
    let value = value.trim();
    let (date, time_with_zone) = value
        .split_once(' ')
        .or_else(|| value.split_once('T'))
        .unwrap_or((value, "00:00:00"));
    let (time, offset_minutes) = split_time_offset(time_with_zone)?;
    let mut date_parts = date.split('-');
    let year = date_parts.next()?.parse::<i32>().ok()?;
    let month = date_parts.next()?.parse::<u32>().ok()?;
    let day = date_parts.next()?.parse::<u32>().ok()?;
    if date_parts.next().is_some() {
        return None;
    }
    let mut time_parts = time.split(':');
    let hour = time_parts.next()?.parse::<u32>().ok()?;
    let minute = time_parts.next().unwrap_or("0").parse::<u32>().ok()?;
    let second = time_parts.next().unwrap_or("0").parse::<u32>().ok()?;
    if time_parts.next().is_some() || !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }
    if hour > 23 || minute > 59 || second > 59 {
        return None;
    }
    let days = days_from_civil(year, month, day)?;
    days.checked_mul(86_400_000)?
        .checked_add(i64::from(hour) * 3_600_000)?
        .checked_add(i64::from(minute) * 60_000)?
        .checked_add(i64::from(second) * 1_000)?
        .checked_sub(offset_minutes * 60_000)
}

fn split_time_offset(time: &str) -> Option<(&str, i64)> {
    let time = time.trim();
    if let Some(time) = time.strip_suffix('Z') {
        return Some((time, 0));
    }
    let offset_start = time
        .char_indices()
        .skip(1)
        .find_map(|(index, ch)| matches!(ch, '+' | '-').then_some(index));
    let Some(offset_start) = offset_start else {
        return Some((time, 0));
    };
    let (time, offset) = time.split_at(offset_start);
    let sign = if offset.starts_with('-') { -1 } else { 1 };
    let offset = &offset[1..];
    let (hours, minutes) = offset.split_once(':')?;
    let hours = hours.parse::<i64>().ok()?;
    let minutes = minutes.parse::<i64>().ok()?;
    if hours > 23 || minutes > 59 {
        return None;
    }
    Some((time, sign * (hours * 60 + minutes)))
}

fn days_from_civil(year: i32, month: u32, day: u32) -> Option<i64> {
    let year = i64::from(year) - i64::from(month <= 2);
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let yoe = year - era * 400;
    let month = i64::from(month);
    let day = i64::from(day);
    let doy = (153 * (month + if month > 2 { -3 } else { 9 }) + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    Some(era * 146097 + doe - 719468)
}

fn evaluate_in_list(
    batch: &RecordBatch,
    column: &str,
    values: &[crate::execution::LiteralValue],
    negated: bool,
    has_null: bool,
) -> Result<BooleanArray> {
    let mut mask = BooleanArray::from(vec![false; batch.num_rows()]);

    for value in values {
        let comparison = ComparisonExpr {
            column: column.to_string(),
            op: ComparisonOp::Eq,
            value: value.clone(),
        };
        let value_mask = evaluate_comparison(batch, &comparison)?;
        mask = or_masks(&mask, &value_mask);
    }

    if has_null {
        mask = nullify_false_values(&mask);
    }
    if negated {
        return Ok(not_mask(&mask));
    }
    Ok(mask)
}

fn evaluate_like(
    batch: &RecordBatch,
    column: &str,
    pattern: &str,
    negated: bool,
    escape: Option<char>,
) -> Result<BooleanArray> {
    let column_index = column_index(batch, column)?;
    let column_array = batch.column(column_index);
    let values = column_array
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| DodamError::UnsupportedFilterType {
            column: column.to_string(),
            data_type: column_array.data_type().clone(),
        })?;
    let fast_pattern = fast_like_pattern(pattern, escape);
    let mut builder = BooleanBuilder::with_capacity(values.len());
    if let Some(fast_pattern) = fast_pattern {
        for row in 0..values.len() {
            if values.is_null(row) {
                builder.append_null();
                continue;
            }
            let matched = fast_pattern.matches(values.value(row));
            builder.append_value(if negated { !matched } else { matched });
        }
    } else {
        let tokens = like_pattern_tokens(pattern, escape)?;
        for row in 0..values.len() {
            if values.is_null(row) {
                builder.append_null();
                continue;
            }
            let matched = like_matches(values.value(row), &tokens);
            builder.append_value(if negated { !matched } else { matched });
        }
    }
    Ok(builder.finish())
}

enum FastLikePattern<'a> {
    All,
    Exact(&'a str),
    Prefix(&'a str),
    Suffix(&'a str),
    Contains(FastLikeSegment<'a>),
    Ordered {
        starts_with_any: bool,
        ends_with_any: bool,
        segments: Vec<FastLikeSegment<'a>>,
    },
}

struct FastLikeSegment<'a> {
    text: &'a str,
    finder: Finder<'a>,
}

impl<'a> FastLikeSegment<'a> {
    fn new(text: &'a str) -> Self {
        Self {
            text,
            finder: Finder::new(text.as_bytes()),
        }
    }
}

impl FastLikePattern<'_> {
    fn matches(&self, value: &str) -> bool {
        match self {
            Self::All => true,
            Self::Exact(segment) => value == *segment,
            Self::Prefix(segment) => value.starts_with(segment),
            Self::Suffix(segment) => value.ends_with(segment),
            Self::Contains(segment) => segment.finder.find(value.as_bytes()).is_some(),
            Self::Ordered {
                starts_with_any,
                ends_with_any,
                segments,
            } => {
                if !starts_with_any && !value.starts_with(segments[0].text) {
                    return false;
                }
                if !ends_with_any && !value.ends_with(segments[segments.len() - 1].text) {
                    return false;
                }

                let mut offset = 0;
                for (index, segment) in segments.iter().enumerate() {
                    if index == 0 && !starts_with_any {
                        offset = segment.text.len();
                        continue;
                    }
                    if index + 1 == segments.len() && !ends_with_any {
                        let suffix_start = value.len().saturating_sub(segment.text.len());
                        if suffix_start < offset {
                            return false;
                        }
                        continue;
                    }
                    let Some(relative) = segment.finder.find(&value.as_bytes()[offset..]) else {
                        return false;
                    };
                    offset += relative + segment.text.len();
                }
                true
            }
        }
    }
}

fn fast_like_pattern(pattern: &str, escape: Option<char>) -> Option<FastLikePattern<'_>> {
    if escape.is_some() || pattern.contains('_') {
        return None;
    }
    let starts_with_any = pattern.starts_with('%');
    let ends_with_any = pattern.ends_with('%');
    let segments = pattern
        .split('%')
        .filter(|segment| !segment.is_empty())
        .collect::<Vec<_>>();
    match segments.as_slice() {
        [] => Some(FastLikePattern::All),
        [segment] if !starts_with_any && !ends_with_any => Some(FastLikePattern::Exact(segment)),
        [segment] if !starts_with_any && ends_with_any => Some(FastLikePattern::Prefix(segment)),
        [segment] if starts_with_any && !ends_with_any => Some(FastLikePattern::Suffix(segment)),
        [segment] => Some(FastLikePattern::Contains(FastLikeSegment::new(segment))),
        _ => Some(FastLikePattern::Ordered {
            starts_with_any,
            ends_with_any,
            segments: segments.into_iter().map(FastLikeSegment::new).collect(),
        }),
    }
}

#[derive(Debug, Clone, Copy)]
enum LikeToken {
    AnyMany,
    AnyOne,
    Literal(char),
}

fn like_pattern_tokens(pattern: &str, escape: Option<char>) -> Result<Vec<LikeToken>> {
    let mut tokens = Vec::new();
    let mut chars = pattern.chars();
    while let Some(ch) = chars.next() {
        if Some(ch) == escape {
            let Some(escaped) = chars.next() else {
                return Err(DodamError::InvalidFilter(format!(
                    "LIKE pattern ends with escape character: {pattern}"
                )));
            };
            tokens.push(LikeToken::Literal(escaped));
        } else if ch == '%' {
            if !matches!(tokens.last(), Some(LikeToken::AnyMany)) {
                tokens.push(LikeToken::AnyMany);
            }
        } else if ch == '_' {
            tokens.push(LikeToken::AnyOne);
        } else {
            tokens.push(LikeToken::Literal(ch));
        }
    }
    Ok(tokens)
}

fn like_matches(value: &str, pattern: &[LikeToken]) -> bool {
    let value = value.chars().collect::<Vec<_>>();
    let mut matched = vec![false; value.len() + 1];
    matched[0] = true;
    for token in pattern {
        let mut next = vec![false; value.len() + 1];
        match token {
            LikeToken::AnyMany => {
                let mut reachable = false;
                for index in 0..=value.len() {
                    reachable |= matched[index];
                    next[index] = reachable;
                }
            }
            LikeToken::AnyOne => {
                for index in 0..value.len() {
                    next[index + 1] = matched[index];
                }
            }
            LikeToken::Literal(expected) => {
                for index in 0..value.len() {
                    next[index + 1] = matched[index] && value[index] == *expected;
                }
            }
        }
        matched = next;
    }
    matched[value.len()]
}

fn nullify_false_values(mask: &BooleanArray) -> BooleanArray {
    let values = (0..mask.len())
        .map(|row| {
            if mask.is_valid(row) && mask.value(row) {
                Some(true)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    BooleanArray::from(values)
}

fn evaluate_is_null(batch: &RecordBatch, column: &str, negated: bool) -> Result<BooleanArray> {
    let column_index = column_index(batch, column)?;
    let column = batch.column(column_index);
    if negated {
        Ok(is_not_null(column.as_ref())?)
    } else {
        Ok(is_null(column.as_ref())?)
    }
}

fn compare<T: arrow::array::Array>(
    column: &ArrayRef,
    scalar: &Scalar<T>,
    op: ComparisonOp,
) -> arrow::error::Result<BooleanArray> {
    match op {
        ComparisonOp::Eq => eq(column, scalar),
        ComparisonOp::NotEq => neq(column, scalar),
        ComparisonOp::Lt => lt(column, scalar),
        ComparisonOp::LtEq => lt_eq(column, scalar),
        ComparisonOp::Gt => gt(column, scalar),
        ComparisonOp::GtEq => gt_eq(column, scalar),
    }
}

fn compare_columns(
    left: &ArrayRef,
    right: &ArrayRef,
    op: ComparisonOp,
) -> arrow::error::Result<BooleanArray> {
    match op {
        ComparisonOp::Eq => eq(left, right),
        ComparisonOp::NotEq => neq(left, right),
        ComparisonOp::Lt => lt(left, right),
        ComparisonOp::LtEq => lt_eq(left, right),
        ComparisonOp::Gt => gt(left, right),
        ComparisonOp::GtEq => gt_eq(left, right),
    }
}

fn compare_decimal128_columns(
    left: &ArrayRef,
    right: &ArrayRef,
    op: ComparisonOp,
) -> Result<BooleanArray> {
    let DataType::Decimal128(_, left_scale) = left.data_type() else {
        unreachable!("validated decimal left column");
    };
    let DataType::Decimal128(_, right_scale) = right.data_type() else {
        unreachable!("validated decimal right column");
    };
    let left = left
        .as_any()
        .downcast_ref::<Decimal128Array>()
        .expect("Decimal128 left column");
    let right = right
        .as_any()
        .downcast_ref::<Decimal128Array>()
        .expect("Decimal128 right column");
    if left.len() != right.len() {
        return Err(DodamError::InvalidFilter(
            "decimal column length mismatch".to_string(),
        ));
    }
    let scale = (*left_scale).max(*right_scale);
    let left_factor = decimal_align_factor(*left_scale, scale)?;
    let right_factor = decimal_align_factor(*right_scale, scale)?;
    Ok(BooleanArray::from(
        (0..left.len())
            .map(|row| {
                if left.is_null(row) || right.is_null(row) {
                    return None;
                }
                let left = left.value(row).checked_mul(left_factor)?;
                let right = right.value(row).checked_mul(right_factor)?;
                Some(compare_i128_values(left, op, right))
            })
            .collect::<Vec<_>>(),
    ))
}

fn decimal_align_factor(from_scale: i8, to_scale: i8) -> Result<i128> {
    if to_scale < from_scale {
        return Err(DodamError::InvalidFilter(format!(
            "cannot align decimal scale {from_scale} to {to_scale}"
        )));
    }
    let scale = usize::try_from(to_scale - from_scale)
        .map_err(|_| DodamError::InvalidFilter("decimal scale is too large".to_string()))?;
    decimal_scale_factor(scale)
}

fn compare_i128_values(left: i128, op: ComparisonOp, right: i128) -> bool {
    match op {
        ComparisonOp::Eq => left == right,
        ComparisonOp::NotEq => left != right,
        ComparisonOp::Lt => left < right,
        ComparisonOp::LtEq => left <= right,
        ComparisonOp::Gt => left > right,
        ComparisonOp::GtEq => left >= right,
    }
}

fn and_masks(left: &BooleanArray, right: &BooleanArray) -> BooleanArray {
    and_kleene(left, right).expect("boolean arrays with equal length")
}

fn or_masks(left: &BooleanArray, right: &BooleanArray) -> BooleanArray {
    or_kleene(left, right).expect("boolean arrays with equal length")
}

fn not_mask(mask: &BooleanArray) -> BooleanArray {
    not(mask).expect("boolean array")
}

fn apply_projection(batch: RecordBatch, projection: &Projection) -> Result<RecordBatch> {
    let Projection::Columns(columns) = projection else {
        return Ok(batch);
    };

    let indices = columns
        .iter()
        .map(|column| column_index(&batch, column))
        .collect::<Result<Vec<_>>>()?;
    Ok(batch.project(&indices)?)
}

pub(crate) fn column_index(batch: &RecordBatch, column: &str) -> Result<usize> {
    batch
        .schema()
        .fields()
        .iter()
        .position(|field| field.name() == column)
        .ok_or_else(|| DodamError::UnknownColumn(column.to_string()))
}

fn output_column_count(projection: &Projection, scanned_columns: usize) -> usize {
    match projection {
        Projection::All => scanned_columns,
        Projection::Columns(columns) => columns.len(),
    }
}

fn elapsed_nanos(start: Instant) -> u64 {
    start.elapsed().as_nanos().min(u64::MAX as u128) as u64
}
