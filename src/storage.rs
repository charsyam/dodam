use std::fs::File;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Instant;
use std::time::SystemTime;

use arrow::datatypes::{DataType, SchemaRef};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::{ArrowReaderMetadata, ParquetRecordBatchReaderBuilder};
use parquet::data_type::{ByteArray, FixedLenByteArray};
use parquet::file::statistics::Statistics;
use std::collections::{BTreeMap, HashMap};

use crate::error::{DodamError, Result};
use crate::execution::{ComparisonExpr, ComparisonOp, Expr, Projection};

#[derive(Debug, Clone)]
pub struct ObjectMetadata {
    pub len: u64,
    pub modified: Option<SystemTime>,
}

pub trait ObjectStore: Send + Sync {
    fn open(&self, path: &Path) -> Result<File>;
    fn metadata(&self, path: &Path) -> Result<ObjectMetadata>;
}

#[derive(Debug, Default)]
pub struct LocalFileSystemObjectStore;

impl ObjectStore for LocalFileSystemObjectStore {
    fn open(&self, path: &Path) -> Result<File> {
        Ok(File::open(path)?)
    }

    fn metadata(&self, path: &Path) -> Result<ObjectMetadata> {
        let metadata = std::fs::metadata(path)?;
        Ok(ObjectMetadata {
            len: metadata.len(),
            modified: metadata.modified().ok(),
        })
    }
}

#[derive(Debug, Clone)]
struct CachedMetadata {
    len: u64,
    modified: Option<SystemTime>,
    metadata: ArrowReaderMetadata,
}

#[derive(Debug, Default)]
pub struct ParquetMetadataCache {
    entries: Mutex<HashMap<PathBuf, CachedMetadata>>,
}

impl ParquetMetadataCache {
    pub fn get(&self, path: impl AsRef<Path>) -> Result<ArrowReaderMetadata> {
        self.get_with_store(path, &LocalFileSystemObjectStore)
    }

    pub fn get_with_store(
        &self,
        path: impl AsRef<Path>,
        store: &dyn ObjectStore,
    ) -> Result<ArrowReaderMetadata> {
        let path = path.as_ref();
        let object_metadata = store.metadata(path)?;
        let len = object_metadata.len;
        let modified = object_metadata.modified;

        {
            let entries = self.entries.lock().expect("metadata cache lock");
            if let Some(entry) = entries.get(path)
                && entry.len == len
                && entry.modified == modified
            {
                return Ok(entry.metadata.clone());
            }
        }

        let file = store.open(path)?;
        let metadata = ArrowReaderMetadata::load(&file, Default::default())?;
        let mut entries = self.entries.lock().expect("metadata cache lock");
        entries.insert(
            path.to_path_buf(),
            CachedMetadata {
                len,
                modified,
                metadata: metadata.clone(),
            },
        );
        Ok(metadata)
    }

    pub fn len(&self) -> usize {
        self.entries.lock().expect("metadata cache lock").len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.lock().expect("metadata cache lock").is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParquetScanTask {
    pub path: PathBuf,
    pub row_group: usize,
    pub partition_values: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ParquetScanTaskPlan {
    pub tasks: Vec<ParquetScanTask>,
    pub row_groups_total: usize,
    pub schema_columns: usize,
    pub projected_columns: usize,
    pub projected_columns_fixed_width: bool,
    pub compressed_bytes_total: u64,
    pub compressed_bytes_scanned: u64,
    pub metadata_nanos: u64,
    pub planning_nanos: u64,
}

#[derive(Debug, Clone)]
pub struct ParquetFileStatistics {
    pub schema: SchemaRef,
    pub rows: usize,
    pub row_groups: usize,
    pub compressed_bytes: u64,
}

pub struct ParquetBatchReader {
    inner: parquet::arrow::arrow_reader::ParquetRecordBatchReader,
    projected_columns: usize,
    row_groups_total: usize,
    row_groups_scanned: usize,
    compressed_bytes_total: u64,
    compressed_bytes_scanned: u64,
    metadata_nanos: u64,
    planning_nanos: u64,
}

impl ParquetBatchReader {
    pub fn try_new(
        path: impl AsRef<Path>,
        batch_size: usize,
        projection: &Projection,
        pruning_predicates: &[Expr],
        metadata_cache: &ParquetMetadataCache,
        store: &dyn ObjectStore,
    ) -> Result<Self> {
        let path = path.as_ref();
        let file = store.open(path)?;
        let metadata_start = Instant::now();
        let metadata = metadata_cache.get_with_store(path, store)?;
        let metadata_nanos = elapsed_nanos(metadata_start);
        let planning_start = Instant::now();
        let builder = ParquetRecordBatchReaderBuilder::new_with_metadata(file, metadata)
            .with_batch_size(batch_size);
        let projected_columns = projected_column_count(builder.schema(), projection);
        let row_groups_total = builder.metadata().num_row_groups();
        let column_indices = projection_indices_for_schema(builder.schema(), projection)?;
        let row_groups = if pruning_predicates.is_empty() {
            None
        } else {
            Some(prune_row_groups(&builder, pruning_predicates)?)
        };
        let row_groups_scanned = row_groups
            .as_ref()
            .map_or(row_groups_total, |row_groups| row_groups.len());
        let all_row_groups = (0..row_groups_total).collect::<Vec<_>>();
        let compressed_bytes_total =
            compressed_bytes_for_row_groups(&builder, &column_indices, &all_row_groups);
        let compressed_bytes_scanned = compressed_bytes_for_row_groups(
            &builder,
            &column_indices,
            row_groups.as_deref().unwrap_or(&all_row_groups),
        );
        let builder = apply_projection(builder, projection)?;
        let builder = if let Some(row_groups) = row_groups {
            builder.with_row_groups(row_groups)
        } else {
            builder
        };
        let planning_nanos = elapsed_nanos(planning_start);
        let inner = builder.build()?;

        Ok(Self {
            inner,
            projected_columns,
            row_groups_total,
            row_groups_scanned,
            compressed_bytes_total,
            compressed_bytes_scanned,
            metadata_nanos,
            planning_nanos,
        })
    }

    pub fn try_new_with_row_groups(
        path: impl AsRef<Path>,
        batch_size: usize,
        projection: &Projection,
        row_groups: Vec<usize>,
        metadata_cache: &ParquetMetadataCache,
        store: &dyn ObjectStore,
    ) -> Result<Self> {
        let path = path.as_ref();
        let file = store.open(path)?;
        let metadata_start = Instant::now();
        let metadata = metadata_cache.get_with_store(path, store)?;
        let metadata_nanos = elapsed_nanos(metadata_start);
        let planning_start = Instant::now();
        let builder = ParquetRecordBatchReaderBuilder::new_with_metadata(file, metadata)
            .with_batch_size(batch_size);
        let projected_columns = projected_column_count(builder.schema(), projection);
        let row_groups_total = builder.metadata().num_row_groups();
        let column_indices = projection_indices_for_schema(builder.schema(), projection)?;
        let row_groups_scanned = row_groups.len();
        let all_row_groups = (0..row_groups_total).collect::<Vec<_>>();
        let compressed_bytes_total =
            compressed_bytes_for_row_groups(&builder, &column_indices, &all_row_groups);
        let compressed_bytes_scanned =
            compressed_bytes_for_row_groups(&builder, &column_indices, &row_groups);
        let builder = apply_projection(builder, projection)?.with_row_groups(row_groups);
        let planning_nanos = elapsed_nanos(planning_start);
        let inner = builder.build()?;

        Ok(Self {
            inner,
            projected_columns,
            row_groups_total,
            row_groups_scanned,
            compressed_bytes_total,
            compressed_bytes_scanned,
            metadata_nanos,
            planning_nanos,
        })
    }

    pub fn projected_columns(&self) -> usize {
        self.projected_columns
    }

    pub fn row_groups_total(&self) -> usize {
        self.row_groups_total
    }

    pub fn row_groups_scanned(&self) -> usize {
        self.row_groups_scanned
    }

    pub fn compressed_bytes_total(&self) -> u64 {
        self.compressed_bytes_total
    }

    pub fn compressed_bytes_scanned(&self) -> u64 {
        self.compressed_bytes_scanned
    }

    pub fn metadata_nanos(&self) -> u64 {
        self.metadata_nanos
    }

    pub fn planning_nanos(&self) -> u64 {
        self.planning_nanos
    }
}

pub fn plan_parquet_scan_tasks(
    path: impl AsRef<Path>,
    projection: &Projection,
    pruning_predicates: &[Expr],
    metadata_cache: &ParquetMetadataCache,
    store: &dyn ObjectStore,
) -> Result<ParquetScanTaskPlan> {
    let path = path.as_ref();
    let file = store.open(path)?;
    let metadata_start = Instant::now();
    let metadata = metadata_cache.get_with_store(path, store)?;
    let metadata_nanos = elapsed_nanos(metadata_start);
    let planning_start = Instant::now();
    let builder = ParquetRecordBatchReaderBuilder::new_with_metadata(file, metadata);
    let row_groups_total = builder.metadata().num_row_groups();
    let schema_columns = builder.schema().fields().len();
    let column_indices = projection_indices_for_schema(builder.schema(), projection)?;
    let projected_columns = column_indices.len();
    let projected_columns_fixed_width = column_indices.iter().all(|index| {
        builder
            .schema()
            .fields()
            .get(*index)
            .is_some_and(|field| is_fixed_width_scan_type(field.data_type()))
    });
    let row_groups = if pruning_predicates.is_empty() {
        (0..row_groups_total).collect()
    } else {
        prune_row_groups(&builder, pruning_predicates)?
    };
    let all_row_groups = (0..row_groups_total).collect::<Vec<_>>();
    let compressed_bytes_total =
        compressed_bytes_for_row_groups(&builder, &column_indices, &all_row_groups);
    let compressed_bytes_scanned =
        compressed_bytes_for_row_groups(&builder, &column_indices, &row_groups);

    let tasks = row_groups
        .into_iter()
        .map(|row_group| ParquetScanTask {
            path: path.to_path_buf(),
            row_group,
            partition_values: BTreeMap::new(),
        })
        .collect();
    let planning_nanos = elapsed_nanos(planning_start);

    Ok(ParquetScanTaskPlan {
        tasks,
        row_groups_total,
        schema_columns,
        projected_columns,
        projected_columns_fixed_width,
        compressed_bytes_total,
        compressed_bytes_scanned,
        metadata_nanos,
        planning_nanos,
    })
}

fn is_fixed_width_scan_type(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Boolean
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float16
            | DataType::Float32
            | DataType::Float64
            | DataType::Date32
            | DataType::Date64
            | DataType::Time32(_)
            | DataType::Time64(_)
            | DataType::Timestamp(_, _)
            | DataType::Duration(_)
    )
}

pub fn read_parquet_file_statistics(
    path: impl AsRef<Path>,
    metadata_cache: &ParquetMetadataCache,
    store: &dyn ObjectStore,
) -> Result<ParquetFileStatistics> {
    let path = path.as_ref();
    let file = store.open(path)?;
    let metadata = metadata_cache.get_with_store(path, store)?;
    let builder = ParquetRecordBatchReaderBuilder::new_with_metadata(file, metadata);
    let row_groups = builder.metadata().num_row_groups();
    let all_row_groups = (0..row_groups).collect::<Vec<_>>();
    let all_columns = (0..builder.schema().fields().len()).collect::<Vec<_>>();
    let compressed_bytes = compressed_bytes_for_row_groups(&builder, &all_columns, &all_row_groups);
    let rows = builder
        .metadata()
        .row_groups()
        .iter()
        .map(|row_group| usize::try_from(row_group.num_rows()).unwrap_or(usize::MAX))
        .fold(0_usize, usize::saturating_add);

    Ok(ParquetFileStatistics {
        schema: builder.schema().clone(),
        rows,
        row_groups,
        compressed_bytes,
    })
}

impl Iterator for ParquetBatchReader {
    type Item = Result<RecordBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|result| result.map_err(Into::into))
    }
}

fn apply_projection(
    builder: ParquetRecordBatchReaderBuilder<File>,
    projection: &Projection,
) -> Result<ParquetRecordBatchReaderBuilder<File>> {
    let Projection::Columns(columns) = projection else {
        return Ok(builder);
    };

    let indices = projection_indices(builder.schema(), columns)?;
    let mask = ProjectionMask::roots(builder.parquet_schema(), indices);
    Ok(builder.with_projection(mask))
}

fn projected_column_count(schema: &arrow::datatypes::SchemaRef, projection: &Projection) -> usize {
    match projection {
        Projection::All => schema.fields().len(),
        Projection::Columns(columns) => columns.len(),
    }
}

fn elapsed_nanos(start: Instant) -> u64 {
    start.elapsed().as_nanos().min(u64::MAX as u128) as u64
}

fn projection_indices(
    schema: &arrow::datatypes::SchemaRef,
    columns: &[String],
) -> Result<Vec<usize>> {
    columns
        .iter()
        .map(|column| {
            schema
                .fields()
                .iter()
                .position(|field| field.name() == column)
                .ok_or_else(|| DodamError::UnknownColumn(column.clone()))
        })
        .collect()
}

fn projection_indices_for_schema(
    schema: &arrow::datatypes::SchemaRef,
    projection: &Projection,
) -> Result<Vec<usize>> {
    match projection {
        Projection::All => Ok((0..schema.fields().len()).collect()),
        Projection::Columns(columns) => projection_indices(schema, columns),
    }
}

fn compressed_bytes_for_row_groups(
    builder: &ParquetRecordBatchReaderBuilder<File>,
    column_indices: &[usize],
    row_groups: &[usize],
) -> u64 {
    row_groups
        .iter()
        .filter_map(|row_group| builder.metadata().row_groups().get(*row_group))
        .flat_map(|row_group| {
            column_indices
                .iter()
                .filter_map(|column| row_group.columns().get(*column))
        })
        .map(|column| column.compressed_size().max(0) as u64)
        .sum()
}

fn prune_row_groups(
    builder: &ParquetRecordBatchReaderBuilder<File>,
    pruning_predicates: &[Expr],
) -> Result<Vec<usize>> {
    let mut row_groups = Vec::new();
    for row_group_index in 0..builder.metadata().row_groups().len() {
        if pruning_predicates
            .iter()
            .map(|predicate| row_group_may_match(builder, row_group_index, predicate))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .all(|may_match| may_match)
        {
            row_groups.push(row_group_index);
        }
    }

    Ok(row_groups)
}

fn row_group_may_match(
    builder: &ParquetRecordBatchReaderBuilder<File>,
    row_group_index: usize,
    expr: &Expr,
) -> Result<bool> {
    match expr {
        Expr::Comparison(comparison) => {
            if matches!(comparison.value, crate::execution::LiteralValue::Null) {
                return Ok(true);
            }
            row_group_may_match_comparison(builder, row_group_index, comparison)
        }
        Expr::And(left, right) => Ok(row_group_may_match(builder, row_group_index, left)?
            && row_group_may_match(builder, row_group_index, right)?),
        Expr::Or(left, right) => Ok(row_group_may_match(builder, row_group_index, left)?
            || row_group_may_match(builder, row_group_index, right)?),
        Expr::Boolean(_)
        | Expr::ColumnComparison { .. }
        | Expr::InList { .. }
        | Expr::Like { .. }
        | Expr::IsNull { .. }
        | Expr::Not(_) => Ok(true),
    }
}

fn row_group_may_match_comparison(
    builder: &ParquetRecordBatchReaderBuilder<File>,
    row_group_index: usize,
    comparison: &ComparisonExpr,
) -> Result<bool> {
    let Some(column_index) = builder
        .schema()
        .fields()
        .iter()
        .position(|field| field.name() == &comparison.column)
    else {
        return Ok(true);
    };
    let data_type = builder.schema().field(column_index).data_type();

    let Some(row_group) = builder.metadata().row_groups().get(row_group_index) else {
        return Ok(true);
    };
    let Some(column) = row_group.columns().get(column_index) else {
        return Ok(true);
    };
    let Some(statistics) = column.statistics() else {
        return Ok(true);
    };

    may_contain_comparison(statistics, data_type, comparison)
}

fn may_contain_comparison(
    statistics: &Statistics,
    data_type: &DataType,
    comparison: &ComparisonExpr,
) -> Result<bool> {
    if statistics.is_min_max_deprecated()
        || !statistics.min_is_exact()
        || !statistics.max_is_exact()
    {
        return Ok(true);
    }

    match (data_type, statistics) {
        (DataType::Boolean, Statistics::Boolean(typed)) => {
            let value = comparison.value.as_bool(&comparison.column)?;
            Ok(comparison_may_match(
                typed.min_opt(),
                typed.max_opt(),
                &value,
                comparison.op,
            ))
        }
        (DataType::Int32, Statistics::Int32(typed)) => {
            let value = comparison.value.as_i32(&comparison.column)?;
            Ok(comparison_may_match(
                typed.min_opt(),
                typed.max_opt(),
                &value,
                comparison.op,
            ))
        }
        (DataType::Int64, Statistics::Int64(typed)) => {
            let value = comparison.value.as_i64(&comparison.column)?;
            Ok(comparison_may_match(
                typed.min_opt(),
                typed.max_opt(),
                &value,
                comparison.op,
            ))
        }
        (DataType::Float64, Statistics::Double(typed)) => {
            let value = comparison.value.as_f64(&comparison.column)?;
            Ok(float_comparison_may_match(
                typed.min_opt(),
                typed.max_opt(),
                value,
                comparison.op,
            ))
        }
        (DataType::Date32, Statistics::Int32(typed)) => {
            let value = date_literal_as_i32(&comparison.value, &comparison.column)?;
            Ok(comparison_may_match(
                typed.min_opt(),
                typed.max_opt(),
                &value,
                comparison.op,
            ))
        }
        (DataType::Date64, Statistics::Int64(typed)) => {
            let value = date_literal_as_i64(&comparison.value, &comparison.column)?;
            Ok(comparison_may_match(
                typed.min_opt(),
                typed.max_opt(),
                &value,
                comparison.op,
            ))
        }
        (DataType::Utf8, Statistics::ByteArray(typed)) => {
            let value = comparison.value.as_str();
            let value = ByteArray::from(value.as_str());
            Ok(bytes_comparison_may_match(
                typed.min_opt(),
                typed.max_opt(),
                value.as_ref(),
                comparison.op,
            ))
        }
        (DataType::Decimal128(_, scale), Statistics::Int32(typed)) => {
            let value = decimal_literal_as_i128(&comparison.value, *scale, &comparison.column)?;
            let min = typed.min_opt().map(|value| i128::from(*value));
            let max = typed.max_opt().map(|value| i128::from(*value));
            Ok(comparison_may_match(
                min.as_ref(),
                max.as_ref(),
                &value,
                comparison.op,
            ))
        }
        (DataType::Decimal128(_, scale), Statistics::Int64(typed)) => {
            let value = decimal_literal_as_i128(&comparison.value, *scale, &comparison.column)?;
            let min = typed.min_opt().map(|value| i128::from(*value));
            let max = typed.max_opt().map(|value| i128::from(*value));
            Ok(comparison_may_match(
                min.as_ref(),
                max.as_ref(),
                &value,
                comparison.op,
            ))
        }
        (DataType::Decimal128(_, scale), Statistics::FixedLenByteArray(typed)) => {
            let value = decimal_literal_as_i128(&comparison.value, *scale, &comparison.column)?;
            let Some(min) = typed.min_opt().and_then(fixed_len_decimal_to_i128) else {
                return Ok(true);
            };
            let Some(max) = typed.max_opt().and_then(fixed_len_decimal_to_i128) else {
                return Ok(true);
            };
            Ok(comparison_may_match(
                Some(&min),
                Some(&max),
                &value,
                comparison.op,
            ))
        }
        (DataType::Decimal128(_, scale), Statistics::ByteArray(typed)) => {
            let value = decimal_literal_as_i128(&comparison.value, *scale, &comparison.column)?;
            let Some(min) = typed
                .min_opt()
                .and_then(|value| decimal_bytes_to_i128(value.as_ref()))
            else {
                return Ok(true);
            };
            let Some(max) = typed
                .max_opt()
                .and_then(|value| decimal_bytes_to_i128(value.as_ref()))
            else {
                return Ok(true);
            };
            Ok(comparison_may_match(
                Some(&min),
                Some(&max),
                &value,
                comparison.op,
            ))
        }
        (DataType::Timestamp(unit, _), Statistics::Int64(typed)) => {
            let value = timestamp_literal_as_i64(&comparison.value, *unit, &comparison.column)?;
            Ok(comparison_may_match(
                typed.min_opt(),
                typed.max_opt(),
                &value,
                comparison.op,
            ))
        }
        _ => Ok(true),
    }
}

fn comparison_may_match<T: Ord>(
    min: Option<&T>,
    max: Option<&T>,
    value: &T,
    op: ComparisonOp,
) -> bool {
    match op {
        ComparisonOp::Eq => {
            min.is_none_or(|min| value >= min) && max.is_none_or(|max| value <= max)
        }
        ComparisonOp::NotEq => true,
        ComparisonOp::Lt => min.is_none_or(|min| min < value),
        ComparisonOp::LtEq => min.is_none_or(|min| min <= value),
        ComparisonOp::Gt => max.is_none_or(|max| max > value),
        ComparisonOp::GtEq => max.is_none_or(|max| max >= value),
    }
}

fn bytes_comparison_may_match(
    min: Option<&ByteArray>,
    max: Option<&ByteArray>,
    value: &[u8],
    op: ComparisonOp,
) -> bool {
    match op {
        ComparisonOp::Eq => {
            min.is_none_or(|min| value >= min.as_ref())
                && max.is_none_or(|max| value <= max.as_ref())
        }
        ComparisonOp::NotEq => true,
        ComparisonOp::Lt => min.is_none_or(|min| min.as_ref() < value),
        ComparisonOp::LtEq => min.is_none_or(|min| min.as_ref() <= value),
        ComparisonOp::Gt => max.is_none_or(|max| max.as_ref() > value),
        ComparisonOp::GtEq => max.is_none_or(|max| max.as_ref() >= value),
    }
}

fn float_comparison_may_match(
    min: Option<&f64>,
    max: Option<&f64>,
    value: f64,
    op: ComparisonOp,
) -> bool {
    if value.is_nan()
        || min.is_some_and(|value| value.is_nan())
        || max.is_some_and(|value| value.is_nan())
    {
        return true;
    }

    match op {
        ComparisonOp::Eq => {
            min.is_none_or(|min| value >= *min) && max.is_none_or(|max| value <= *max)
        }
        ComparisonOp::NotEq => true,
        ComparisonOp::Lt => min.is_none_or(|min| *min < value),
        ComparisonOp::LtEq => min.is_none_or(|min| *min <= value),
        ComparisonOp::Gt => max.is_none_or(|max| *max > value),
        ComparisonOp::GtEq => max.is_none_or(|max| *max >= value),
    }
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

fn fixed_len_decimal_to_i128(value: &FixedLenByteArray) -> Option<i128> {
    decimal_bytes_to_i128(value.as_ref())
}

fn decimal_bytes_to_i128(value: &[u8]) -> Option<i128> {
    if value.is_empty() || value.len() > 16 {
        return None;
    }
    let sign = if value[0] & 0x80 == 0 { 0 } else { 0xff };
    let mut bytes = [sign; 16];
    bytes[16 - value.len()..].copy_from_slice(value);
    Some(i128::from_be_bytes(bytes))
}

fn timestamp_literal_as_i64(
    value: &crate::execution::LiteralValue,
    unit: arrow::datatypes::TimeUnit,
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
        arrow::datatypes::TimeUnit::Second => Ok(millis / 1_000),
        arrow::datatypes::TimeUnit::Millisecond => Ok(millis),
        arrow::datatypes::TimeUnit::Microsecond => millis
            .checked_mul(1_000)
            .ok_or_else(|| DodamError::InvalidFilter(format!("{column}={value}"))),
        arrow::datatypes::TimeUnit::Nanosecond => millis
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
