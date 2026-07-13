use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;

use arrow::array::{ArrayRef, Int32Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use crate::error::{DodamError, Result};

pub struct SendableBatchStream {
    inner: Box<dyn Iterator<Item = Result<RecordBatch>> + Send>,
    scan_plan_metrics: Arc<ScanPlanMetricsCounter>,
}

pub trait RecordBatchSink {
    fn write_batch(&mut self, batch: &RecordBatch) -> Result<()>;

    fn write_primitive_batch(&mut self, batch: PrimitiveBatch) -> Result<bool> {
        self.write_batch(&batch.into_record_batch()?)?;
        Ok(true)
    }

    fn supports_i32_utf8_rows(&self) -> bool {
        false
    }

    fn write_i32_utf8_rows(
        &mut self,
        _left: &Int32Array,
        _left_indices: &[u32],
        _right_arrays: &[&StringArray],
        _right_batch_indices: &[usize],
        _right_row_indices: &[u32],
    ) -> Result<bool> {
        Ok(false)
    }

    fn supports_i32_rows(&self) -> bool {
        false
    }

    fn write_i32_rows(&mut self, _array: &Int32Array, _indices: &[u32]) -> Result<bool> {
        Ok(false)
    }

    fn discards_output(&self) -> bool {
        false
    }

    fn finish(&mut self) -> Result<()> {
        Ok(())
    }
}

#[derive(Debug)]
pub struct PrimitiveBatch {
    pub columns: Vec<PrimitiveColumn>,
}

#[derive(Debug)]
pub struct PrimitiveColumn {
    pub name: String,
    pub data_type: DataType,
    pub nullable: bool,
    pub values: PrimitiveColumnValues,
}

#[derive(Debug)]
pub enum PrimitiveColumnValues {
    I32(Vec<i32>),
    I64(Vec<i64>),
}

impl PrimitiveBatch {
    pub fn num_rows(&self) -> usize {
        self.columns
            .first()
            .map(|column| column.values.len())
            .unwrap_or_default()
    }

    pub fn is_empty(&self) -> bool {
        self.num_rows() == 0
    }

    pub fn slice(self, offset: usize, len: usize) -> Result<Self> {
        let row_count = self.num_rows();
        if offset > row_count || offset + len > row_count {
            return Err(DodamError::UnsupportedSql(
                "primitive batch slice is out of bounds".to_string(),
            ));
        }
        let columns = self
            .columns
            .into_iter()
            .map(|column| PrimitiveColumn {
                name: column.name,
                data_type: column.data_type,
                nullable: column.nullable,
                values: column.values.slice(offset, len),
            })
            .collect();
        Ok(Self { columns })
    }

    pub fn concat(batches: Vec<Self>) -> Result<Self> {
        let mut iter = batches.into_iter().filter(|batch| !batch.is_empty());
        let Some(mut output) = iter.next() else {
            return Ok(Self {
                columns: Vec::new(),
            });
        };
        let remaining = iter.collect::<Vec<_>>();
        for batch in &remaining {
            if batch.columns.len() != output.columns.len() {
                return Err(DodamError::UnsupportedSql(
                    "primitive batch column count mismatch".to_string(),
                ));
            }
            for (target, source) in output.columns.iter().zip(batch.columns.iter()) {
                if target.name != source.name || target.data_type != source.data_type {
                    return Err(DodamError::UnsupportedSql(
                        "primitive batch schema mismatch".to_string(),
                    ));
                }
            }
        }
        for column_index in 0..output.columns.len() {
            let additional = remaining
                .iter()
                .map(|batch| batch.columns[column_index].values.len())
                .sum();
            output.columns[column_index].values.reserve(additional);
        }
        for batch in remaining {
            for (target, source) in output.columns.iter_mut().zip(batch.columns.into_iter()) {
                target.values.extend(source.values)?;
            }
        }
        Ok(output)
    }

    pub fn into_record_batch(self) -> Result<RecordBatch> {
        let schema = self.schema();
        self.into_record_batch_with_schema(schema)
    }

    pub fn into_record_batch_with_schema(self, schema: Arc<Schema>) -> Result<RecordBatch> {
        Ok(RecordBatch::try_new(schema, self.into_arrays())?)
    }

    pub fn schema(&self) -> Arc<Schema> {
        let fields = self
            .columns
            .iter()
            .map(|column| Field::new(&column.name, column.data_type.clone(), column.nullable))
            .collect::<Vec<_>>();
        Arc::new(Schema::new(fields))
    }

    fn into_arrays(self) -> Vec<ArrayRef> {
        self.columns
            .into_iter()
            .map(|column| column.values.into_array())
            .collect::<Vec<_>>()
    }

    pub fn matches_schema(&self, schema: &Schema) -> bool {
        schema.fields().len() == self.columns.len()
            && schema
                .fields()
                .iter()
                .zip(self.columns.iter())
                .all(|(field, column)| {
                    field.name() == &column.name
                        && field.data_type() == &column.data_type
                        && field.is_nullable() == column.nullable
                })
    }
}

impl PrimitiveColumnValues {
    pub fn len(&self) -> usize {
        match self {
            Self::I32(values) => values.len(),
            Self::I64(values) => values.len(),
        }
    }

    pub fn slice(self, offset: usize, len: usize) -> Self {
        match self {
            Self::I32(values) => Self::I32(values[offset..offset + len].to_vec()),
            Self::I64(values) => Self::I64(values[offset..offset + len].to_vec()),
        }
    }

    pub fn extend(&mut self, other: Self) -> Result<()> {
        match (self, other) {
            (Self::I32(target), Self::I32(source)) => target.extend(source),
            (Self::I64(target), Self::I64(source)) => target.extend(source),
            _ => {
                return Err(DodamError::UnsupportedSql(
                    "primitive batch value type mismatch".to_string(),
                ));
            }
        }
        Ok(())
    }

    pub fn reserve(&mut self, additional: usize) {
        match self {
            Self::I32(values) => values.reserve(additional),
            Self::I64(values) => values.reserve(additional),
        }
    }

    pub fn into_array(self) -> ArrayRef {
        match self {
            Self::I32(values) => std::sync::Arc::new(Int32Array::from(values)) as ArrayRef,
            Self::I64(values) => std::sync::Arc::new(Int64Array::from(values)) as ArrayRef,
        }
    }
}

pub fn write_stream_to_sink(
    mut stream: SendableBatchStream,
    sink: &mut dyn RecordBatchSink,
) -> Result<ScanPlanMetrics> {
    for batch in stream.by_ref() {
        let batch = batch?;
        sink.write_batch(&batch)?;
    }
    sink.finish()?;
    Ok(stream.into_scan_plan_metrics())
}

impl SendableBatchStream {
    pub fn new(
        inner: Box<dyn Iterator<Item = Result<RecordBatch>> + Send>,
        scan_plan_metrics: Arc<ScanPlanMetricsCounter>,
    ) -> Self {
        Self {
            inner,
            scan_plan_metrics,
        }
    }

    pub fn empty() -> Self {
        Self::new(Box::new(std::iter::empty()), Arc::default())
    }

    pub fn from_batches(batches: Vec<RecordBatch>) -> Self {
        Self::new(Box::new(batches.into_iter().map(Ok)), Arc::default())
    }

    pub fn scan_plan_metrics(&self) -> ScanPlanMetrics {
        self.scan_plan_metrics.snapshot()
    }

    pub fn into_scan_plan_metrics(self) -> ScanPlanMetrics {
        let Self {
            inner,
            scan_plan_metrics,
        } = self;
        drop(inner);
        scan_plan_metrics.snapshot()
    }

    pub(crate) fn into_parts(
        self,
    ) -> (
        Box<dyn Iterator<Item = Result<RecordBatch>> + Send>,
        Arc<ScanPlanMetricsCounter>,
    ) {
        (self.inner, self.scan_plan_metrics)
    }
}

impl Iterator for SendableBatchStream {
    type Item = Result<RecordBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ScanPlanMetrics {
    pub row_groups_total: usize,
    pub row_groups_scanned: usize,
    pub row_groups_pruned: usize,
    pub compressed_bytes_total: u64,
    pub compressed_bytes_scanned: u64,
    pub compressed_bytes_pruned: u64,
    pub metadata_nanos: u64,
    pub planning_nanos: u64,
    pub decode_nanos: u64,
    pub filter_nanos: u64,
    pub projection_nanos: u64,
    pub limit_nanos: u64,
    pub parquet_next_calls: usize,
    pub parquet_eof_calls: usize,
    pub parquet_output_batches: usize,
    pub parquet_output_rows: usize,
    pub parquet_zero_row_batches: usize,
    pub parquet_next_nanos: u64,
    pub parquet_max_next_nanos: u64,
    pub join_build_rows: usize,
    pub join_probe_rows: usize,
    pub join_output_rows: usize,
    pub join_spill_files: usize,
    pub join_spill_bytes: u64,
    pub join_repartitions: usize,
    pub join_heavy_hitters: usize,
    pub join_bloom_filtered_rows: usize,
    pub join_nested_loop_fallbacks: usize,
    pub join_peak_build_bytes: u64,
    pub join_build_nanos: u64,
    pub join_materialize_nanos: u64,
}

#[derive(Debug, Default)]
pub struct ScanPlanMetricsCounter {
    row_groups_total: AtomicUsize,
    row_groups_scanned: AtomicUsize,
    compressed_bytes_total: AtomicUsize,
    compressed_bytes_scanned: AtomicUsize,
    metadata_nanos: AtomicU64,
    planning_nanos: AtomicU64,
    decode_nanos: AtomicU64,
    filter_nanos: AtomicU64,
    projection_nanos: AtomicU64,
    limit_nanos: AtomicU64,
    parquet_next_calls: AtomicUsize,
    parquet_eof_calls: AtomicUsize,
    parquet_output_batches: AtomicUsize,
    parquet_output_rows: AtomicUsize,
    parquet_zero_row_batches: AtomicUsize,
    parquet_next_nanos: AtomicU64,
    parquet_max_next_nanos: AtomicU64,
    join_build_rows: AtomicUsize,
    join_probe_rows: AtomicUsize,
    join_output_rows: AtomicUsize,
    join_spill_files: AtomicUsize,
    join_spill_bytes: AtomicUsize,
    join_repartitions: AtomicUsize,
    join_heavy_hitters: AtomicUsize,
    join_bloom_filtered_rows: AtomicUsize,
    join_nested_loop_fallbacks: AtomicUsize,
    join_peak_build_bytes: AtomicUsize,
    join_build_nanos: AtomicU64,
    join_materialize_nanos: AtomicU64,
}

impl ScanPlanMetricsCounter {
    pub(crate) fn add_scan_plan(
        &self,
        row_groups_total: usize,
        row_groups_scanned: usize,
        compressed_bytes_total: u64,
        compressed_bytes_scanned: u64,
    ) {
        self.row_groups_total
            .fetch_add(row_groups_total, Ordering::Relaxed);
        self.row_groups_scanned
            .fetch_add(row_groups_scanned, Ordering::Relaxed);
        self.compressed_bytes_total
            .fetch_add(compressed_bytes_total as usize, Ordering::Relaxed);
        self.compressed_bytes_scanned
            .fetch_add(compressed_bytes_scanned as usize, Ordering::Relaxed);
    }

    pub(crate) fn add_metadata_time(&self, elapsed: Duration) {
        self.metadata_nanos.fetch_add(
            elapsed.as_nanos().min(u64::MAX as u128) as u64,
            Ordering::Relaxed,
        );
    }

    pub(crate) fn add_planning_time(&self, elapsed: Duration) {
        self.planning_nanos.fetch_add(
            elapsed.as_nanos().min(u64::MAX as u128) as u64,
            Ordering::Relaxed,
        );
    }

    pub(crate) fn add_decode_time(&self, elapsed: Duration) {
        self.decode_nanos.fetch_add(
            elapsed.as_nanos().min(u64::MAX as u128) as u64,
            Ordering::Relaxed,
        );
    }

    pub(crate) fn add_filter_time(&self, elapsed: Duration) {
        self.filter_nanos.fetch_add(
            elapsed.as_nanos().min(u64::MAX as u128) as u64,
            Ordering::Relaxed,
        );
    }

    pub(crate) fn add_projection_time(&self, elapsed: Duration) {
        self.projection_nanos.fetch_add(
            elapsed.as_nanos().min(u64::MAX as u128) as u64,
            Ordering::Relaxed,
        );
    }

    pub(crate) fn add_limit_time(&self, elapsed: Duration) {
        self.limit_nanos.fetch_add(
            elapsed.as_nanos().min(u64::MAX as u128) as u64,
            Ordering::Relaxed,
        );
    }

    pub(crate) fn add_parquet_reader_stats(
        &self,
        next_calls: usize,
        eof_calls: usize,
        output_batches: usize,
        output_rows: usize,
        zero_row_batches: usize,
        next_nanos: u64,
        max_next_nanos: u64,
    ) {
        self.parquet_next_calls
            .fetch_add(next_calls, Ordering::Relaxed);
        self.parquet_eof_calls
            .fetch_add(eof_calls, Ordering::Relaxed);
        self.parquet_output_batches
            .fetch_add(output_batches, Ordering::Relaxed);
        self.parquet_output_rows
            .fetch_add(output_rows, Ordering::Relaxed);
        self.parquet_zero_row_batches
            .fetch_add(zero_row_batches, Ordering::Relaxed);
        self.parquet_next_nanos
            .fetch_add(next_nanos, Ordering::Relaxed);
        let mut current = self.parquet_max_next_nanos.load(Ordering::Relaxed);
        while max_next_nanos > current {
            match self.parquet_max_next_nanos.compare_exchange_weak(
                current,
                max_next_nanos,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(observed) => current = observed,
            }
        }
    }

    pub(crate) fn add_join_build_rows(&self, rows: usize) {
        self.join_build_rows.fetch_add(rows, Ordering::Relaxed);
    }

    pub(crate) fn add_join_build_time(&self, elapsed: Duration) {
        self.join_build_nanos.fetch_add(
            elapsed.as_nanos().min(u64::MAX as u128) as u64,
            Ordering::Relaxed,
        );
    }

    pub(crate) fn add_join_probe_rows(&self, rows: usize) {
        self.join_probe_rows.fetch_add(rows, Ordering::Relaxed);
    }

    pub(crate) fn add_join_output_rows(&self, rows: usize) {
        self.join_output_rows.fetch_add(rows, Ordering::Relaxed);
    }

    pub(crate) fn add_join_materialize_time(&self, elapsed: Duration) {
        self.join_materialize_nanos.fetch_add(
            elapsed.as_nanos().min(u64::MAX as u128) as u64,
            Ordering::Relaxed,
        );
    }

    pub(crate) fn add_join_spill_file(&self, bytes: u64) {
        self.join_spill_files.fetch_add(1, Ordering::Relaxed);
        self.join_spill_bytes
            .fetch_add(bytes as usize, Ordering::Relaxed);
    }

    pub(crate) fn add_join_repartition(&self) {
        self.join_repartitions.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn add_join_heavy_hitters(&self, keys: usize) {
        self.join_heavy_hitters.fetch_add(keys, Ordering::Relaxed);
    }

    pub(crate) fn add_join_bloom_filtered_rows(&self, rows: usize) {
        self.join_bloom_filtered_rows
            .fetch_add(rows, Ordering::Relaxed);
    }

    pub(crate) fn add_join_nested_loop_fallback(&self) {
        self.join_nested_loop_fallbacks
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn observe_join_build_bytes(&self, bytes: u64) {
        let bytes = bytes.min(usize::MAX as u64) as usize;
        let mut current = self.join_peak_build_bytes.load(Ordering::Relaxed);
        while bytes > current {
            match self.join_peak_build_bytes.compare_exchange_weak(
                current,
                bytes,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(observed) => current = observed,
            }
        }
    }

    pub(crate) fn snapshot(&self) -> ScanPlanMetrics {
        let row_groups_total = self.row_groups_total.load(Ordering::Relaxed);
        let row_groups_scanned = self.row_groups_scanned.load(Ordering::Relaxed);
        let compressed_bytes_total = self.compressed_bytes_total.load(Ordering::Relaxed) as u64;
        let compressed_bytes_scanned = self.compressed_bytes_scanned.load(Ordering::Relaxed) as u64;
        ScanPlanMetrics {
            row_groups_total,
            row_groups_scanned,
            row_groups_pruned: row_groups_total.saturating_sub(row_groups_scanned),
            compressed_bytes_total,
            compressed_bytes_scanned,
            compressed_bytes_pruned: compressed_bytes_total
                .saturating_sub(compressed_bytes_scanned),
            metadata_nanos: self.metadata_nanos.load(Ordering::Relaxed),
            planning_nanos: self.planning_nanos.load(Ordering::Relaxed),
            decode_nanos: self.decode_nanos.load(Ordering::Relaxed),
            filter_nanos: self.filter_nanos.load(Ordering::Relaxed),
            projection_nanos: self.projection_nanos.load(Ordering::Relaxed),
            limit_nanos: self.limit_nanos.load(Ordering::Relaxed),
            parquet_next_calls: self.parquet_next_calls.load(Ordering::Relaxed),
            parquet_eof_calls: self.parquet_eof_calls.load(Ordering::Relaxed),
            parquet_output_batches: self.parquet_output_batches.load(Ordering::Relaxed),
            parquet_output_rows: self.parquet_output_rows.load(Ordering::Relaxed),
            parquet_zero_row_batches: self.parquet_zero_row_batches.load(Ordering::Relaxed),
            parquet_next_nanos: self.parquet_next_nanos.load(Ordering::Relaxed),
            parquet_max_next_nanos: self.parquet_max_next_nanos.load(Ordering::Relaxed),
            join_build_rows: self.join_build_rows.load(Ordering::Relaxed),
            join_probe_rows: self.join_probe_rows.load(Ordering::Relaxed),
            join_output_rows: self.join_output_rows.load(Ordering::Relaxed),
            join_spill_files: self.join_spill_files.load(Ordering::Relaxed),
            join_spill_bytes: self.join_spill_bytes.load(Ordering::Relaxed) as u64,
            join_repartitions: self.join_repartitions.load(Ordering::Relaxed),
            join_heavy_hitters: self.join_heavy_hitters.load(Ordering::Relaxed),
            join_bloom_filtered_rows: self.join_bloom_filtered_rows.load(Ordering::Relaxed),
            join_nested_loop_fallbacks: self.join_nested_loop_fallbacks.load(Ordering::Relaxed),
            join_peak_build_bytes: self.join_peak_build_bytes.load(Ordering::Relaxed) as u64,
            join_build_nanos: self.join_build_nanos.load(Ordering::Relaxed),
            join_materialize_nanos: self.join_materialize_nanos.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ScanMetrics {
    pub fragments: usize,
    pub batches: usize,
    pub rows: usize,
    pub columns: usize,
    pub row_groups_total: usize,
    pub row_groups_scanned: usize,
    pub row_groups_pruned: usize,
    pub compressed_bytes_total: u64,
    pub compressed_bytes_scanned: u64,
    pub compressed_bytes_pruned: u64,
    pub metadata_nanos: u64,
    pub planning_nanos: u64,
    pub decode_nanos: u64,
    pub filter_nanos: u64,
    pub projection_nanos: u64,
    pub limit_nanos: u64,
    pub join_build_rows: usize,
    pub join_probe_rows: usize,
    pub join_output_rows: usize,
    pub join_spill_files: usize,
    pub join_spill_bytes: u64,
    pub join_repartitions: usize,
    pub join_heavy_hitters: usize,
    pub join_bloom_filtered_rows: usize,
    pub join_nested_loop_fallbacks: usize,
    pub join_peak_build_bytes: u64,
}
