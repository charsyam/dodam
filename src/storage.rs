use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::time::Instant;
use std::time::SystemTime;

use arrow::array::{Array, BooleanArray, BooleanBuilder, Int32Array, Int64Array};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::error::ArrowError;
use arrow::record_batch::RecordBatch;
use bytes::Bytes;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::{
    ArrowPredicate, ArrowPredicateFn, ArrowReaderMetadata, ArrowReaderOptions,
    ParquetRecordBatchReaderBuilder, RowFilter, RowSelection,
};
use parquet::basic::{Encoding, Type as ParquetPhysicalType};
use parquet::column::page::{Page, PageReader};
use parquet::column::reader::{ColumnReader, ColumnReaderImpl};
use parquet::data_type::{ByteArray, ByteArrayType, FixedLenByteArray, Int32Type, Int64Type};
use parquet::errors::{ParquetError, Result as ParquetResult};
use parquet::file::metadata::PageIndexPolicy;
use parquet::file::reader::{ChunkReader, FileReader as ParquetFileReader, Length};
use parquet::file::serialized_reader::SerializedFileReader;
use parquet::file::statistics::Statistics;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::hash::{Hash, Hasher};

use crate::cost::{
    FragmentedSelectedPayloadCostInput, SelectedPayloadCostInput,
    choose_fragmented_selected_payload_full_decode, choose_selected_payload,
};
use crate::error::{DodamError, Result};
use crate::execution::{
    ComparisonExpr, ComparisonOp, Expr, FilterExpr, Projection, evaluate_filter_mask,
};
use crate::vector::{DictionaryStringValues, RawColumnView, SelectionRuns, SelectionRunsBuilder};

mod selected_i64_decoder;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrimitiveRowGroupMinMax {
    pub row_group: usize,
    pub rows: usize,
    pub null_count: Option<u64>,
    pub min: i128,
    pub max: i128,
}

pub(crate) struct DirectI32I64DecimalI32SelectedBatch<'a> {
    pub(crate) keys: &'a [i32],
    pub(crate) sums: &'a [i64],
    pub(crate) decimals: &'a [i64],
    pub(crate) dates: &'a [i32],
    pub(crate) predicate_applied: bool,
}

#[derive(Clone, Copy)]
pub(crate) struct SelectedI64ChunkRange {
    pub(crate) selected_offset: usize,
    pub(crate) values_offset: usize,
    pub(crate) len: usize,
}

pub(crate) enum DirectI32I32DictionaryI64SelectedBatch<'a> {
    Compact {
        first: &'a [i32],
        second: &'a [i32],
        dictionary_ids: &'a [i32],
        dictionary: &'a [Bytes],
        sums: &'a [i64],
    },
    Masked {
        first: &'a [i32],
        second: &'a [i32],
        dictionary_ids: &'a [i32],
        dictionary: &'a [Bytes],
        sums: &'a [i64],
        selection: SelectionRuns<'a>,
    },
    SumChunkRanges {
        first: &'a [i32],
        second: &'a [i32],
        dictionary_ids: &'a [i32],
        dictionary: &'a [Bytes],
        chunks: &'a [SelectedI64ChunkRange],
        sums: &'a [i64],
        selected_rows: usize,
    },
}

impl<'a> DirectI32I32DictionaryI64SelectedBatch<'a> {
    pub(crate) fn compact(
        first: &'a [i32],
        second: &'a [i32],
        dictionary_ids: &'a [i32],
        dictionary: &'a [Bytes],
        sums: &'a [i64],
    ) -> Self {
        Self::Compact {
            first,
            second,
            dictionary_ids,
            dictionary,
            sums,
        }
    }

    pub(crate) fn first(&self) -> &'a [i32] {
        match self {
            Self::Compact { first, .. } | Self::Masked { first, .. } => first,
            Self::SumChunkRanges { first, .. } => first,
        }
    }

    pub(crate) fn second(&self) -> &'a [i32] {
        match self {
            Self::Compact { second, .. } | Self::Masked { second, .. } => second,
            Self::SumChunkRanges { second, .. } => second,
        }
    }

    pub(crate) fn dictionary_ids(&self) -> &'a [i32] {
        match self {
            Self::Compact { dictionary_ids, .. } | Self::Masked { dictionary_ids, .. } => {
                dictionary_ids
            }
            Self::SumChunkRanges { dictionary_ids, .. } => dictionary_ids,
        }
    }

    pub(crate) fn dictionary(&self) -> &'a [Bytes] {
        match self {
            Self::Compact { dictionary, .. } | Self::Masked { dictionary, .. } => dictionary,
            Self::SumChunkRanges { dictionary, .. } => dictionary,
        }
    }

    pub(crate) fn sums(&self) -> &'a [i64] {
        match self {
            Self::Compact { sums, .. } | Self::Masked { sums, .. } => sums,
            Self::SumChunkRanges { .. } => &[],
        }
    }

    pub(crate) fn selection(&self) -> Option<SelectionRuns<'a>> {
        match self {
            Self::Compact { .. } => None,
            Self::Masked { selection, .. } => Some(*selection),
            Self::SumChunkRanges { .. } => None,
        }
    }

    pub(crate) fn sum_chunk_ranges(
        &self,
    ) -> Option<(&'a [SelectedI64ChunkRange], &'a [i64], usize)> {
        match self {
            Self::SumChunkRanges {
                chunks,
                sums,
                selected_rows,
                ..
            } => Some((*chunks, *sums, *selected_rows)),
            _ => None,
        }
    }
}

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
    page_index: bool,
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
        let page_index = parquet_page_index_enabled();

        {
            let entries = self.entries.lock().expect("metadata cache lock");
            if let Some(entry) = entries.get(path)
                && entry.len == len
                && entry.modified == modified
                && entry.page_index == page_index
            {
                return Ok(entry.metadata.clone());
            }
        }

        let file = store.open(path)?;
        let metadata = ArrowReaderMetadata::load(&file, arrow_reader_options())?;
        let mut entries = self.entries.lock().expect("metadata cache lock");
        entries.insert(
            path.to_path_buf(),
            CachedMetadata {
                len,
                modified,
                page_index,
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

fn parquet_page_index_enabled() -> bool {
    !std::env::var("DODAM_PARQUET_PAGE_INDEX")
        .is_ok_and(|value| matches!(value.as_str(), "0" | "false" | "FALSE" | "no" | "NO"))
}

fn arrow_reader_options() -> ArrowReaderOptions {
    if parquet_page_index_enabled() {
        ArrowReaderOptions::new().with_page_index_policy(PageIndexPolicy::Optional)
    } else {
        ArrowReaderOptions::new()
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct CachedFileChunkKey {
    path: PathBuf,
    len: u64,
    modified: Option<SystemTime>,
    chunk_start: u64,
}

#[derive(Debug, Default)]
struct ParquetFileCacheShard {
    entries: RwLock<HashMap<CachedFileChunkKey, Arc<CachedFileChunkEntry>>>,
    admissions: Mutex<HashMap<CachedFileChunkKey, u64>>,
}

#[derive(Debug)]
struct CachedFileChunkEntry {
    bytes: Bytes,
    last_access_epoch: AtomicU64,
}

#[derive(Debug)]
pub struct ParquetFileCache {
    max_bytes: usize,
    chunk_bytes: usize,
    admit_on_first_read: bool,
    shards: Vec<ParquetFileCacheShard>,
    bytes: AtomicU64,
    access_epoch: AtomicU64,
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
    read_bytes: AtomicU64,
    deferred_admissions: AtomicU64,
}

const PARQUET_FILE_CACHE_SHARDS: usize = 64;
const DEFAULT_PARQUET_FILE_CACHE_BYTES: usize = 0;
const DEFAULT_PARQUET_FILE_CACHE_CHUNK_BYTES: usize = 512 * 1024;

impl Default for ParquetFileCache {
    fn default() -> Self {
        Self::from_env()
    }
}

impl ParquetFileCache {
    pub fn disabled() -> Self {
        Self {
            max_bytes: 0,
            chunk_bytes: DEFAULT_PARQUET_FILE_CACHE_CHUNK_BYTES,
            admit_on_first_read: false,
            shards: (0..PARQUET_FILE_CACHE_SHARDS)
                .map(|_| ParquetFileCacheShard::default())
                .collect(),
            bytes: AtomicU64::new(0),
            access_epoch: AtomicU64::new(1),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            read_bytes: AtomicU64::new(0),
            deferred_admissions: AtomicU64::new(0),
        }
    }

    pub fn from_env() -> Self {
        let max_bytes = std::env::var("DODAM_FILE_CACHE_BYTES")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(DEFAULT_PARQUET_FILE_CACHE_BYTES);
        let chunk_bytes = std::env::var("DODAM_FILE_CACHE_CHUNK_BYTES")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(DEFAULT_PARQUET_FILE_CACHE_CHUNK_BYTES);
        Self {
            max_bytes,
            chunk_bytes,
            admit_on_first_read: true,
            shards: (0..PARQUET_FILE_CACHE_SHARDS)
                .map(|_| ParquetFileCacheShard::default())
                .collect(),
            bytes: AtomicU64::new(0),
            access_epoch: AtomicU64::new(1),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            read_bytes: AtomicU64::new(0),
            deferred_admissions: AtomicU64::new(0),
        }
    }

    fn enabled(&self) -> bool {
        self.max_bytes > 0
    }

    fn chunk_bytes(&self) -> usize {
        self.chunk_bytes
    }

    fn get_or_read_range(
        &self,
        path: impl AsRef<Path>,
        len: u64,
        modified: Option<SystemTime>,
        file: &File,
        start: u64,
        length: usize,
    ) -> ParquetResult<Bytes> {
        if self.max_bytes == 0 {
            return read_file_range(file, start, length);
        }
        let chunk_start = (start / self.chunk_bytes as u64) * self.chunk_bytes as u64;
        let local_start = (start - chunk_start) as usize;
        let chunk_len = self.chunk_len(len, chunk_start);
        let key = CachedFileChunkKey {
            path: path.as_ref().to_path_buf(),
            len,
            modified,
            chunk_start,
        };
        let shard_index = self.shard_index(&key);
        let epoch = self.next_epoch();

        {
            let entries = self.shards[shard_index]
                .entries
                .read()
                .expect("file cache shard read lock");
            if let Some(entry) = entries.get(&key) {
                entry.last_access_epoch.store(epoch, Ordering::Relaxed);
                self.hits.fetch_add(1, Ordering::Relaxed);
                return Ok(entry.bytes.slice(local_start..local_start + length));
            }
        }

        if chunk_len > self.max_bytes {
            return read_file_range(file, start, length);
        }
        self.misses.fetch_add(1, Ordering::Relaxed);

        let mut entries = self.shards[shard_index]
            .entries
            .write()
            .expect("file cache shard write lock");
        if let Some(entry) = entries.get(&key) {
            entry.last_access_epoch.store(epoch, Ordering::Relaxed);
            self.hits.fetch_add(1, Ordering::Relaxed);
            return Ok(entry.bytes.slice(local_start..local_start + length));
        }

        let entry_size = chunk_len as u64;
        if !self.should_admit(&key, shard_index, epoch, entry_size) {
            let bytes = read_file_range(file, start, length)?;
            self.read_bytes
                .fetch_add(bytes.len() as u64, Ordering::Relaxed);
            return Ok(bytes);
        }
        let bytes = read_file_range(file, chunk_start, chunk_len)?;
        self.read_bytes
            .fetch_add(bytes.len() as u64, Ordering::Relaxed);
        self.evict_until_has_space(entry_size, shard_index, &mut entries);
        entries.insert(
            key,
            Arc::new(CachedFileChunkEntry {
                bytes: bytes.clone(),
                last_access_epoch: AtomicU64::new(epoch),
            }),
        );
        self.bytes.fetch_add(entry_size, Ordering::Relaxed);
        Ok(bytes.slice(local_start..local_start + length))
    }

    fn should_admit(
        &self,
        key: &CachedFileChunkKey,
        shard_index: usize,
        epoch: u64,
        entry_size: u64,
    ) -> bool {
        if self.admit_on_first_read && !self.cache_pressure(entry_size) {
            return true;
        }

        let mut admissions = self.shards[shard_index]
            .admissions
            .lock()
            .expect("file cache admission lock");
        let previous = admissions.remove(key);
        if previous.is_some() {
            return true;
        }
        if admissions.len() >= 4096
            && let Some(oldest_key) = admissions
                .iter()
                .min_by_key(|(_, seen_epoch)| **seen_epoch)
                .map(|(key, _)| key.clone())
        {
            admissions.remove(&oldest_key);
        }
        admissions.insert(key.clone(), epoch);
        self.deferred_admissions.fetch_add(1, Ordering::Relaxed);
        false
    }

    fn cache_pressure(&self, incoming_bytes: u64) -> bool {
        self.bytes
            .load(Ordering::Relaxed)
            .saturating_add(incoming_bytes)
            .saturating_mul(4)
            >= (self.max_bytes as u64).saturating_mul(3)
    }

    fn evict_until_has_space(
        &self,
        incoming_bytes: u64,
        locked_shard_index: usize,
        locked_entries: &mut HashMap<CachedFileChunkKey, Arc<CachedFileChunkEntry>>,
    ) {
        let max_bytes = self.max_bytes as u64;
        let start = self.access_epoch.load(Ordering::Relaxed) as usize;
        let mut attempts = 0usize;
        while self
            .bytes
            .load(Ordering::Relaxed)
            .saturating_add(incoming_bytes)
            > max_bytes
            && attempts < PARQUET_FILE_CACHE_SHARDS * 4
        {
            let shard_index = (start + attempts) % PARQUET_FILE_CACHE_SHARDS;
            if shard_index == locked_shard_index {
                let _ = self.evict_one_from_entries(locked_entries);
            } else if let Ok(mut entries) = self.shards[shard_index].entries.try_write() {
                let _ = self.evict_one_from_entries(&mut entries);
            }
            attempts += 1;
        }
    }

    fn evict_one_from_entries(
        &self,
        entries: &mut HashMap<CachedFileChunkKey, Arc<CachedFileChunkEntry>>,
    ) -> bool {
        let Some(victim_key) = entries
            .iter()
            .min_by_key(|(_, entry)| entry.last_access_epoch.load(Ordering::Relaxed))
            .map(|(key, _)| key.clone())
        else {
            return false;
        };
        if let Some(victim) = entries.remove(&victim_key) {
            self.bytes
                .fetch_sub(victim.bytes.len() as u64, Ordering::Relaxed);
            self.evictions.fetch_add(1, Ordering::Relaxed);
            return true;
        }
        false
    }

    fn shard_index(&self, key: &CachedFileChunkKey) -> usize {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        key.hash(&mut hasher);
        (hasher.finish() as usize) % self.shards.len()
    }

    fn next_epoch(&self) -> u64 {
        self.access_epoch.fetch_add(1, Ordering::Relaxed)
    }

    fn chunk_len(&self, file_len: u64, chunk_start: u64) -> usize {
        file_len
            .saturating_sub(chunk_start)
            .min(self.chunk_bytes as u64) as usize
    }

    pub fn len(&self) -> usize {
        self.shards
            .iter()
            .map(|shard| {
                shard
                    .entries
                    .read()
                    .expect("file cache shard read lock")
                    .len()
            })
            .sum()
    }

    pub fn bytes(&self) -> usize {
        self.bytes.load(Ordering::Relaxed) as usize
    }

    pub fn stats(&self) -> ParquetFileCacheStats {
        ParquetFileCacheStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            read_bytes: self.read_bytes.load(Ordering::Relaxed),
            deferred_admissions: self.deferred_admissions.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ParquetFileCacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub read_bytes: u64,
    pub deferred_admissions: u64,
}

struct CachedParquetChunkReader {
    path: PathBuf,
    len: u64,
    modified: Option<SystemTime>,
    file: File,
    cache: Arc<ParquetFileCache>,
}

impl CachedParquetChunkReader {
    fn new(
        path: impl AsRef<Path>,
        store: &dyn ObjectStore,
        cache: Arc<ParquetFileCache>,
    ) -> Result<Self> {
        let path = path.as_ref();
        let metadata = store.metadata(path)?;
        let file = store.open(path)?;
        Ok(Self {
            path: path.to_path_buf(),
            len: metadata.len,
            modified: metadata.modified,
            file,
            cache,
        })
    }
}

impl Length for CachedParquetChunkReader {
    fn len(&self) -> u64 {
        self.len
    }
}

impl ChunkReader for CachedParquetChunkReader {
    type T = BufReader<File>;

    fn get_read(&self, start: u64) -> ParquetResult<Self::T> {
        let mut reader = self.file.try_clone()?;
        reader.seek(SeekFrom::Start(start))?;
        Ok(BufReader::new(reader))
    }

    fn get_bytes(&self, start: u64, length: usize) -> ParquetResult<Bytes> {
        if start > self.len || start.saturating_add(length as u64) > self.len {
            return Err(ParquetError::EOF(format!(
                "Expected to read {length} bytes at offset {start}, while file has length {}",
                self.len
            )));
        }
        if length == 0 {
            return Ok(Bytes::new());
        }

        let chunk_bytes = self.cache.chunk_bytes() as u64;
        let request_end = start + length as u64;
        let mut chunk_start = (start / chunk_bytes) * chunk_bytes;
        if request_end <= chunk_start.saturating_add(chunk_bytes) {
            return self.cache.get_or_read_range(
                &self.path,
                self.len,
                self.modified,
                &self.file,
                start,
                length,
            );
        }

        let mut output = Vec::with_capacity(length);
        while chunk_start < request_end {
            let chunk_end = chunk_start
                .saturating_add(chunk_bytes)
                .min(self.len)
                .min(request_end);
            let copy_start = start.max(chunk_start);
            let segment_len = (chunk_end - copy_start) as usize;
            let chunk = self.cache.get_or_read_range(
                &self.path,
                self.len,
                self.modified,
                &self.file,
                copy_start,
                segment_len,
            )?;
            output.extend_from_slice(&chunk);
            chunk_start = chunk_start.saturating_add(chunk_bytes);
        }
        Ok(Bytes::from(output))
    }
}

fn read_file_range(file: &File, start: u64, length: usize) -> ParquetResult<Bytes> {
    let mut buffer = Vec::with_capacity(length);
    let mut reader = file.try_clone()?;
    reader.seek(SeekFrom::Start(start))?;
    let read = reader.take(length as u64).read_to_end(&mut buffer)?;
    if read != length {
        return Err(ParquetError::EOF(format!(
            "Expected to read {length} bytes at offset {start}, read only {read}"
        )));
    }
    Ok(Bytes::from(buffer))
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
    projection_order: Option<ProjectionOrderState>,
    projected_columns: usize,
    row_groups_total: usize,
    row_groups_scanned: usize,
    compressed_bytes_total: u64,
    compressed_bytes_scanned: u64,
    metadata_nanos: u64,
    planning_nanos: u64,
    next_calls: usize,
    eof_calls: usize,
    output_batches: usize,
    output_rows: usize,
    zero_row_batches: usize,
    next_nanos: u64,
    max_next_nanos: u64,
    next_samples: Option<Vec<u64>>,
}

enum ProjectionOrderState {
    Pending(Vec<String>),
    Ready(Option<Vec<usize>>),
}

impl ParquetBatchReader {
    pub fn try_new(
        path: impl AsRef<Path>,
        batch_size: usize,
        projection: &Projection,
        pruning_predicates: &[Expr],
        row_filter_predicates: &[Expr],
        metadata_cache: &ParquetMetadataCache,
        file_cache: Arc<ParquetFileCache>,
        store: &dyn ObjectStore,
    ) -> Result<Self> {
        let path = path.as_ref();
        let metadata_start = Instant::now();
        let metadata = metadata_cache.get_with_store(path, store)?;
        let metadata_nanos = elapsed_nanos(metadata_start);
        if file_cache.enabled() {
            let reader = CachedParquetChunkReader::new(path, store, file_cache)?;
            Self::build(
                path,
                reader,
                metadata,
                metadata_nanos,
                batch_size,
                projection,
                pruning_predicates,
                row_filter_predicates,
                None,
            )
        } else {
            let file = store.open(path)?;
            Self::build(
                path,
                file,
                metadata,
                metadata_nanos,
                batch_size,
                projection,
                pruning_predicates,
                row_filter_predicates,
                None,
            )
        }
    }

    pub fn try_new_with_row_groups(
        path: impl AsRef<Path>,
        batch_size: usize,
        projection: &Projection,
        row_groups: Vec<usize>,
        metadata_cache: &ParquetMetadataCache,
        file_cache: Arc<ParquetFileCache>,
        store: &dyn ObjectStore,
    ) -> Result<Self> {
        let path = path.as_ref();
        let metadata_start = Instant::now();
        let metadata = metadata_cache.get_with_store(path, store)?;
        let metadata_nanos = elapsed_nanos(metadata_start);
        if file_cache.enabled() {
            let reader = CachedParquetChunkReader::new(path, store, file_cache)?;
            Self::build(
                path,
                reader,
                metadata,
                metadata_nanos,
                batch_size,
                projection,
                &[],
                &[],
                Some(row_groups),
            )
        } else {
            let file = store.open(path)?;
            Self::build(
                path,
                file,
                metadata,
                metadata_nanos,
                batch_size,
                projection,
                &[],
                &[],
                Some(row_groups),
            )
        }
    }

    pub fn try_new_with_row_groups_dictionary_columns(
        path: impl AsRef<Path>,
        batch_size: usize,
        projection: &Projection,
        row_groups: Vec<usize>,
        dictionary_columns: &[String],
        metadata_cache: &ParquetMetadataCache,
        file_cache: Arc<ParquetFileCache>,
        store: &dyn ObjectStore,
    ) -> Result<Self> {
        let path = path.as_ref();
        let metadata_start = Instant::now();
        let metadata = metadata_cache.get_with_store(path, store)?;
        let metadata_nanos = elapsed_nanos(metadata_start);
        let metadata = metadata_with_dictionary_columns(metadata, dictionary_columns)?;
        if file_cache.enabled() {
            let reader = CachedParquetChunkReader::new(path, store, file_cache)?;
            Self::build(
                path,
                reader,
                metadata,
                metadata_nanos,
                batch_size,
                projection,
                &[],
                &[],
                Some(row_groups),
            )
        } else {
            let file = store.open(path)?;
            Self::build(
                path,
                file,
                metadata,
                metadata_nanos,
                batch_size,
                projection,
                &[],
                &[],
                Some(row_groups),
            )
        }
    }

    pub fn try_new_with_row_groups_selection(
        path: impl AsRef<Path>,
        batch_size: usize,
        projection: &Projection,
        row_groups: Vec<usize>,
        row_selection: RowSelection,
        metadata_cache: &ParquetMetadataCache,
        file_cache: Arc<ParquetFileCache>,
        store: &dyn ObjectStore,
    ) -> Result<Self> {
        let path = path.as_ref();
        let metadata_start = Instant::now();
        let metadata = metadata_cache.get_with_store(path, store)?;
        let metadata_nanos = elapsed_nanos(metadata_start);
        if file_cache.enabled() {
            let reader = CachedParquetChunkReader::new(path, store, file_cache)?;
            Self::build_with_row_selection(
                path,
                reader,
                metadata,
                metadata_nanos,
                batch_size,
                projection,
                row_groups,
                row_selection,
            )
        } else {
            let file = store.open(path)?;
            Self::build_with_row_selection(
                path,
                file,
                metadata,
                metadata_nanos,
                batch_size,
                projection,
                row_groups,
                row_selection,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn try_new_with_row_groups_selection_dictionary_columns(
        path: impl AsRef<Path>,
        batch_size: usize,
        projection: &Projection,
        row_groups: Vec<usize>,
        row_selection: RowSelection,
        dictionary_columns: &[String],
        metadata_cache: &ParquetMetadataCache,
        file_cache: Arc<ParquetFileCache>,
        store: &dyn ObjectStore,
    ) -> Result<Self> {
        let path = path.as_ref();
        let metadata_start = Instant::now();
        let metadata = metadata_cache.get_with_store(path, store)?;
        let metadata_nanos = elapsed_nanos(metadata_start);
        let metadata = metadata_with_dictionary_columns(metadata, dictionary_columns)?;
        if file_cache.enabled() {
            let reader = CachedParquetChunkReader::new(path, store, file_cache)?;
            Self::build_with_row_selection(
                path,
                reader,
                metadata,
                metadata_nanos,
                batch_size,
                projection,
                row_groups,
                row_selection,
            )
        } else {
            let file = store.open(path)?;
            Self::build_with_row_selection(
                path,
                file,
                metadata,
                metadata_nanos,
                batch_size,
                projection,
                row_groups,
                row_selection,
            )
        }
    }

    pub fn try_new_with_row_groups_filtered(
        path: impl AsRef<Path>,
        batch_size: usize,
        projection: &Projection,
        row_groups: Vec<usize>,
        row_filter_predicates: &[Expr],
        metadata_cache: &ParquetMetadataCache,
        file_cache: Arc<ParquetFileCache>,
        store: &dyn ObjectStore,
    ) -> Result<Self> {
        let path = path.as_ref();
        let metadata_start = Instant::now();
        let metadata = metadata_cache.get_with_store(path, store)?;
        let metadata_nanos = elapsed_nanos(metadata_start);
        if file_cache.enabled() {
            let reader = CachedParquetChunkReader::new(path, store, file_cache)?;
            Self::build(
                path,
                reader,
                metadata,
                metadata_nanos,
                batch_size,
                projection,
                &[],
                row_filter_predicates,
                Some(row_groups),
            )
        } else {
            let file = store.open(path)?;
            Self::build(
                path,
                file,
                metadata,
                metadata_nanos,
                batch_size,
                projection,
                &[],
                row_filter_predicates,
                Some(row_groups),
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn try_new_with_row_groups_i64_set_filter(
        path: impl AsRef<Path>,
        batch_size: usize,
        projection: &Projection,
        row_groups: Vec<usize>,
        filter_column: &str,
        keys: Arc<HashSet<i64>>,
        metadata_cache: &ParquetMetadataCache,
        file_cache: Arc<ParquetFileCache>,
        store: &dyn ObjectStore,
    ) -> Result<Self> {
        let path = path.as_ref();
        let metadata_start = Instant::now();
        let metadata = metadata_cache.get_with_store(path, store)?;
        let metadata_nanos = elapsed_nanos(metadata_start);
        if file_cache.enabled() {
            let reader = CachedParquetChunkReader::new(path, store, file_cache)?;
            Self::build_i64_set_filter(
                path,
                reader,
                metadata,
                metadata_nanos,
                batch_size,
                projection,
                row_groups,
                filter_column,
                keys,
            )
        } else {
            let file = store.open(path)?;
            Self::build_i64_set_filter(
                path,
                file,
                metadata,
                metadata_nanos,
                batch_size,
                projection,
                row_groups,
                filter_column,
                keys,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn try_new_with_row_groups_i64_bloom_filter(
        path: impl AsRef<Path>,
        batch_size: usize,
        projection: &Projection,
        row_groups: Vec<usize>,
        filter_column: &str,
        bloom: Arc<I64BloomPredicate>,
        metadata_cache: &ParquetMetadataCache,
        file_cache: Arc<ParquetFileCache>,
        store: &dyn ObjectStore,
    ) -> Result<Self> {
        let path = path.as_ref();
        let metadata_start = Instant::now();
        let metadata = metadata_cache.get_with_store(path, store)?;
        let metadata_nanos = elapsed_nanos(metadata_start);
        if file_cache.enabled() {
            let reader = CachedParquetChunkReader::new(path, store, file_cache)?;
            Self::build_i64_bloom_filter(
                path,
                reader,
                metadata,
                metadata_nanos,
                batch_size,
                projection,
                row_groups,
                filter_column,
                bloom,
            )
        } else {
            let file = store.open(path)?;
            Self::build_i64_bloom_filter(
                path,
                file,
                metadata,
                metadata_nanos,
                batch_size,
                projection,
                row_groups,
                filter_column,
                bloom,
            )
        }
    }

    fn build<T: ChunkReader + 'static>(
        path: &Path,
        input: T,
        metadata: ArrowReaderMetadata,
        metadata_nanos: u64,
        batch_size: usize,
        projection: &Projection,
        pruning_predicates: &[Expr],
        row_filter_predicates: &[Expr],
        requested_row_groups: Option<Vec<usize>>,
    ) -> Result<Self> {
        let planning_start = Instant::now();
        let builder = ParquetRecordBatchReaderBuilder::new_with_metadata(input, metadata)
            .with_batch_size(batch_size);
        let projected_columns = projected_column_count(builder.schema(), projection);
        let row_groups_total = builder.metadata().num_row_groups();
        let column_indices = projection_indices_for_schema(builder.schema(), projection)?;
        let row_groups = if let Some(row_groups) = requested_row_groups {
            Some(row_groups)
        } else if pruning_predicates.is_empty() {
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
        let scanned_row_groups = row_groups.as_deref().unwrap_or(&all_row_groups);
        if row_groups.is_none()
            || scanned_row_groups.len() == all_row_groups.len()
            || parquet_column_chunk_profile_enabled()
        {
            maybe_profile_parquet_projected_columns(
                path,
                &builder,
                &column_indices,
                &all_row_groups,
                scanned_row_groups,
            );
        }
        let builder = apply_projection(builder, projection)?;
        let builder = if let Some(row_groups) = row_groups {
            builder.with_row_groups(row_groups)
        } else {
            builder
        };
        let builder = if let Some(row_filter) = row_filter(&builder, row_filter_predicates)? {
            builder.with_row_filter(row_filter)
        } else {
            builder
        };
        let planning_nanos = elapsed_nanos(planning_start);
        let inner = builder.build()?;

        Ok(Self {
            inner,
            projection_order: projection_order(projection),
            projected_columns,
            row_groups_total,
            row_groups_scanned,
            compressed_bytes_total,
            compressed_bytes_scanned,
            metadata_nanos,
            planning_nanos,
            next_calls: 0,
            eof_calls: 0,
            output_batches: 0,
            output_rows: 0,
            zero_row_batches: 0,
            next_nanos: 0,
            max_next_nanos: 0,
            next_samples: parquet_next_sample_enabled().then(Vec::new),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn build_i64_set_filter<T: ChunkReader + 'static>(
        path: &Path,
        input: T,
        metadata: ArrowReaderMetadata,
        metadata_nanos: u64,
        batch_size: usize,
        projection: &Projection,
        row_groups: Vec<usize>,
        filter_column: &str,
        keys: Arc<HashSet<i64>>,
    ) -> Result<Self> {
        let planning_start = Instant::now();
        let builder = ParquetRecordBatchReaderBuilder::new_with_metadata(input, metadata)
            .with_batch_size(batch_size);
        let projected_columns = projected_column_count(builder.schema(), projection);
        let row_groups_total = builder.metadata().num_row_groups();
        let column_indices = projection_indices_for_schema(builder.schema(), projection)?;
        let all_row_groups = (0..row_groups_total).collect::<Vec<_>>();
        let compressed_bytes_total =
            compressed_bytes_for_row_groups(&builder, &column_indices, &all_row_groups);
        let compressed_bytes_scanned =
            compressed_bytes_for_row_groups(&builder, &column_indices, &row_groups);
        if row_groups.len() == all_row_groups.len() || parquet_column_chunk_profile_enabled() {
            maybe_profile_parquet_projected_columns(
                path,
                &builder,
                &column_indices,
                &all_row_groups,
                &row_groups,
            );
        }
        let filter_index = projection_indices(builder.schema(), &[filter_column.to_string()])?[0];
        let filter_mask = ProjectionMask::roots(builder.parquet_schema(), [filter_index]);
        let keys = I64SetPredicate::from_hash_set(keys.as_ref());
        let filter = ArrowPredicateFn::new(filter_mask, move |batch: RecordBatch| {
            let Some(values) = batch.column(0).as_any().downcast_ref::<Int64Array>() else {
                return Err(ArrowError::ComputeError(
                    "i64 set row filter requires Int64Array".to_string(),
                ));
            };
            Ok(keys.evaluate(values))
        });
        let builder = apply_projection(builder, projection)?
            .with_row_groups(row_groups.clone())
            .with_row_filter(RowFilter::new(vec![Box::new(filter)]));
        let planning_nanos = elapsed_nanos(planning_start);
        let inner = builder.build()?;
        Ok(Self {
            inner,
            projection_order: projection_order(projection),
            projected_columns,
            row_groups_total,
            row_groups_scanned: row_groups.len(),
            compressed_bytes_total,
            compressed_bytes_scanned,
            metadata_nanos,
            planning_nanos,
            next_calls: 0,
            eof_calls: 0,
            output_batches: 0,
            output_rows: 0,
            zero_row_batches: 0,
            next_nanos: 0,
            max_next_nanos: 0,
            next_samples: parquet_next_sample_enabled().then(Vec::new),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn build_i64_bloom_filter<T: ChunkReader + 'static>(
        path: &Path,
        input: T,
        metadata: ArrowReaderMetadata,
        metadata_nanos: u64,
        batch_size: usize,
        projection: &Projection,
        row_groups: Vec<usize>,
        filter_column: &str,
        bloom: Arc<I64BloomPredicate>,
    ) -> Result<Self> {
        let planning_start = Instant::now();
        let builder = ParquetRecordBatchReaderBuilder::new_with_metadata(input, metadata)
            .with_batch_size(batch_size);
        let projected_columns = projected_column_count(builder.schema(), projection);
        let row_groups_total = builder.metadata().num_row_groups();
        let column_indices = projection_indices_for_schema(builder.schema(), projection)?;
        let all_row_groups = (0..row_groups_total).collect::<Vec<_>>();
        let compressed_bytes_total =
            compressed_bytes_for_row_groups(&builder, &column_indices, &all_row_groups);
        let compressed_bytes_scanned =
            compressed_bytes_for_row_groups(&builder, &column_indices, &row_groups);
        if row_groups.len() == all_row_groups.len() || parquet_column_chunk_profile_enabled() {
            maybe_profile_parquet_projected_columns(
                path,
                &builder,
                &column_indices,
                &all_row_groups,
                &row_groups,
            );
        }
        let filter_index = projection_indices(builder.schema(), &[filter_column.to_string()])?[0];
        let filter_mask = ProjectionMask::roots(builder.parquet_schema(), [filter_index]);
        let filter = ArrowPredicateFn::new(filter_mask, move |batch: RecordBatch| {
            let Some(values) = batch.column(0).as_any().downcast_ref::<Int64Array>() else {
                return Err(ArrowError::ComputeError(
                    "i64 bloom row filter requires Int64Array".to_string(),
                ));
            };
            Ok(bloom.evaluate(values))
        });
        let builder = apply_projection(builder, projection)?
            .with_row_groups(row_groups.clone())
            .with_row_filter(RowFilter::new(vec![Box::new(filter)]));
        let planning_nanos = elapsed_nanos(planning_start);
        let inner = builder.build()?;
        Ok(Self {
            inner,
            projection_order: projection_order(projection),
            projected_columns,
            row_groups_total,
            row_groups_scanned: row_groups.len(),
            compressed_bytes_total,
            compressed_bytes_scanned,
            metadata_nanos,
            planning_nanos,
            next_calls: 0,
            eof_calls: 0,
            output_batches: 0,
            output_rows: 0,
            zero_row_batches: 0,
            next_nanos: 0,
            max_next_nanos: 0,
            next_samples: parquet_next_sample_enabled().then(Vec::new),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn build_with_row_selection<T: ChunkReader + 'static>(
        path: &Path,
        input: T,
        metadata: ArrowReaderMetadata,
        metadata_nanos: u64,
        batch_size: usize,
        projection: &Projection,
        row_groups: Vec<usize>,
        row_selection: RowSelection,
    ) -> Result<Self> {
        let planning_start = Instant::now();
        let builder = ParquetRecordBatchReaderBuilder::new_with_metadata(input, metadata)
            .with_batch_size(batch_size);
        let projected_columns = projected_column_count(builder.schema(), projection);
        let row_groups_total = builder.metadata().num_row_groups();
        let column_indices = projection_indices_for_schema(builder.schema(), projection)?;
        let all_row_groups = (0..row_groups_total).collect::<Vec<_>>();
        let compressed_bytes_total =
            compressed_bytes_for_row_groups(&builder, &column_indices, &all_row_groups);
        let compressed_bytes_scanned =
            compressed_bytes_for_row_groups(&builder, &column_indices, &row_groups);
        if row_groups.len() == all_row_groups.len() || parquet_column_chunk_profile_enabled() {
            maybe_profile_parquet_projected_columns(
                path,
                &builder,
                &column_indices,
                &all_row_groups,
                &row_groups,
            );
        }
        let row_groups_scanned = row_groups.len();
        let inner = apply_projection(builder, projection)?
            .with_row_groups(row_groups)
            .with_row_selection(row_selection)
            .build()?;
        let planning_nanos = elapsed_nanos(planning_start);
        Ok(Self {
            inner,
            projection_order: projection_order(projection),
            projected_columns,
            row_groups_total,
            row_groups_scanned,
            compressed_bytes_total,
            compressed_bytes_scanned,
            metadata_nanos,
            planning_nanos,
            next_calls: 0,
            eof_calls: 0,
            output_batches: 0,
            output_rows: 0,
            zero_row_batches: 0,
            next_nanos: 0,
            max_next_nanos: 0,
            next_samples: parquet_next_sample_enabled().then(Vec::new),
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

    pub fn next_calls(&self) -> usize {
        self.next_calls
    }

    pub fn eof_calls(&self) -> usize {
        self.eof_calls
    }

    pub fn output_batches(&self) -> usize {
        self.output_batches
    }

    pub fn output_rows(&self) -> usize {
        self.output_rows
    }

    pub fn zero_row_batches(&self) -> usize {
        self.zero_row_batches
    }

    pub fn next_nanos(&self) -> u64 {
        self.next_nanos
    }

    pub fn max_next_nanos(&self) -> u64 {
        self.max_next_nanos
    }

    pub fn p95_next_nanos(&self) -> u64 {
        percentile_nanos(self.next_samples.as_deref().unwrap_or(&[]), 95)
    }
}

fn parquet_next_sample_enabled() -> bool {
    std::env::var_os("DODAM_SCAN_PROFILE").is_some()
        || std::env::var_os("DODAM_TPCH_PROFILE").is_some()
        || std::env::var_os("DODAM_PROFILE_ORDERED_SINK").is_some()
}

fn percentile_nanos(samples: &[u64], percentile: usize) -> u64 {
    if samples.is_empty() {
        return 0;
    }
    let mut values = samples.to_vec();
    values.sort_unstable();
    let index = values
        .len()
        .saturating_sub(1)
        .saturating_mul(percentile.min(100))
        / 100;
    values[index]
}

#[derive(Debug, Default, Clone)]
pub(crate) struct DirectPrimitiveColumnScanMetrics {
    pub row_groups: usize,
    pub batches: usize,
    pub rows: usize,
    pub read_nanos: u64,
    pub consume_nanos: u64,
    pub column_read_nanos: Vec<u64>,
    pub selected_rows: usize,
    pub selected_runs: usize,
    pub full_payload_batches: usize,
    pub selected_payload_batches: usize,
    pub selected_skip_calls: usize,
    pub selected_read_calls: usize,
    pub selected_skipped_rows: usize,
    pub selected_read_rows: usize,
    pub selected_predicate_nanos: u64,
    pub selected_payload_nanos: u64,
    pub selected_dictionary_nanos: u64,
}

pub(crate) type DirectI64I32I32ScanMetrics = DirectPrimitiveColumnScanMetrics;

pub(crate) enum DirectPrimitiveCountSumPageBatch<'a> {
    I32I64 {
        keys: &'a [u8],
        sums: &'a [u8],
        rows: usize,
    },
    I32NullableI64 {
        keys: &'a [u8],
        key_def_levels: &'a [i16],
        sums: &'a [u8],
        rows: usize,
    },
    I64I64 {
        keys: &'a [u8],
        sums: &'a [u8],
        rows: usize,
    },
}

pub(crate) enum DirectOrderedPrimitiveColumnValues {
    I32(Vec<i32>),
    I64(Vec<i64>),
}

impl DirectOrderedPrimitiveColumnValues {
    pub(crate) fn len(&self) -> usize {
        match self {
            Self::I32(values) => values.len(),
            Self::I64(values) => values.len(),
        }
    }
}

pub(crate) struct DirectOrderedPrimitiveBatch {
    pub(crate) columns: Vec<DirectOrderedPrimitiveColumnValues>,
}

pub(crate) enum DirectSelectedPrimitiveColumnPageView<'a> {
    I32Plain {
        bytes: &'a [u8],
        records: usize,
    },
    I64Plain {
        bytes: &'a [u8],
        records: usize,
    },
    I32Dictionary {
        ids: &'a [i32],
        dictionary: &'a [i32],
        rows_read: usize,
    },
}

impl DirectSelectedPrimitiveColumnPageView<'_> {
    pub(crate) fn value_i128(&self, row: usize) -> Option<i128> {
        match self {
            Self::I32Plain { bytes, records } => {
                (row < *records).then(|| read_i32_le_unchecked(bytes, row) as i128)
            }
            Self::I64Plain { bytes, records } => {
                (row < *records).then(|| read_i64_le_unchecked(bytes, row) as i128)
            }
            Self::I32Dictionary {
                ids,
                dictionary,
                rows_read,
            } => {
                let id = *ids.get(rows_read.saturating_add(row))?;
                let id = usize::try_from(id).ok()?;
                dictionary.get(id).copied().map(i128::from)
            }
        }
    }

    pub(crate) fn value_i32(&self, row: usize) -> Option<i32> {
        match self {
            Self::I32Plain { bytes, records } => {
                (row < *records).then(|| read_i32_le_unchecked(bytes, row))
            }
            Self::I32Dictionary {
                ids,
                dictionary,
                rows_read,
            } => {
                let id = *ids.get(rows_read.saturating_add(row))?;
                let id = usize::try_from(id).ok()?;
                dictionary.get(id).copied()
            }
            Self::I64Plain { .. } => None,
        }
    }

    pub(crate) fn value_i64(&self, row: usize) -> Option<i64> {
        match self {
            Self::I64Plain { bytes, records } => {
                (row < *records).then(|| read_i64_le_unchecked(bytes, row))
            }
            Self::I32Plain { .. } | Self::I32Dictionary { .. } => None,
        }
    }
}

pub(crate) struct DirectSelectedPrimitivePageBatch<'a> {
    pub(crate) columns: Vec<DirectSelectedPrimitiveColumnPageView<'a>>,
    pub(crate) selected_positions: &'a [usize],
}

impl DirectPrimitiveColumnScanMetrics {
    pub(crate) fn merge_from(&mut self, other: Self) {
        self.row_groups = self.row_groups.saturating_add(other.row_groups);
        self.batches = self.batches.saturating_add(other.batches);
        self.rows = self.rows.saturating_add(other.rows);
        self.read_nanos = self.read_nanos.saturating_add(other.read_nanos);
        self.consume_nanos = self.consume_nanos.saturating_add(other.consume_nanos);
        self.selected_rows = self.selected_rows.saturating_add(other.selected_rows);
        self.selected_runs = self.selected_runs.saturating_add(other.selected_runs);
        self.full_payload_batches = self
            .full_payload_batches
            .saturating_add(other.full_payload_batches);
        self.selected_payload_batches = self
            .selected_payload_batches
            .saturating_add(other.selected_payload_batches);
        self.selected_skip_calls = self
            .selected_skip_calls
            .saturating_add(other.selected_skip_calls);
        self.selected_read_calls = self
            .selected_read_calls
            .saturating_add(other.selected_read_calls);
        self.selected_skipped_rows = self
            .selected_skipped_rows
            .saturating_add(other.selected_skipped_rows);
        self.selected_read_rows = self
            .selected_read_rows
            .saturating_add(other.selected_read_rows);
        self.selected_predicate_nanos = self
            .selected_predicate_nanos
            .saturating_add(other.selected_predicate_nanos);
        self.selected_payload_nanos = self
            .selected_payload_nanos
            .saturating_add(other.selected_payload_nanos);
        self.selected_dictionary_nanos = self
            .selected_dictionary_nanos
            .saturating_add(other.selected_dictionary_nanos);
        if self.column_read_nanos.len() < other.column_read_nanos.len() {
            self.column_read_nanos
                .resize(other.column_read_nanos.len(), 0);
        }
        for (index, nanos) in other.column_read_nanos.iter().enumerate() {
            self.column_read_nanos[index] = self.column_read_nanos[index].saturating_add(*nanos);
        }
    }

    fn add_read_nanos(&mut self, nanos: u64) {
        self.read_nanos = self.read_nanos.saturating_add(nanos);
    }

    fn add_consume_nanos(&mut self, nanos: u64) {
        self.consume_nanos = self.consume_nanos.saturating_add(nanos);
    }

    fn add_column_read_nanos(&mut self, index: usize, nanos: u64) {
        if let Some(value) = self.column_read_nanos.get_mut(index) {
            *value = value.saturating_add(nanos);
        }
    }

    fn add_selected_predicate_nanos(&mut self, nanos: u64) {
        self.selected_predicate_nanos = self.selected_predicate_nanos.saturating_add(nanos);
    }

    fn add_selected_payload_nanos(&mut self, nanos: u64) {
        self.selected_payload_nanos = self.selected_payload_nanos.saturating_add(nanos);
    }

    fn add_selected_dictionary_nanos(&mut self, nanos: u64) {
        self.selected_dictionary_nanos = self.selected_dictionary_nanos.saturating_add(nanos);
    }

    fn add_selected_skip(&mut self, rows: usize) {
        self.selected_skip_calls = self.selected_skip_calls.saturating_add(1);
        self.selected_skipped_rows = self.selected_skipped_rows.saturating_add(rows);
    }

    fn add_selected_read(&mut self, rows: usize) {
        self.selected_read_calls = self.selected_read_calls.saturating_add(1);
        self.selected_read_rows = self.selected_read_rows.saturating_add(rows);
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum DirectPrimitiveColumnType {
    I64,
    #[allow(dead_code)]
    I32,
    Date32,
    #[allow(dead_code)]
    Decimal128Int64 {
        precision: u8,
        scale: i8,
    },
    Decimal128Int64Raw {
        precision: u8,
        scale: i8,
    },
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct DirectPrimitiveColumnSpec<'a> {
    pub name: &'a str,
    pub column_type: DirectPrimitiveColumnType,
}

pub(crate) fn parquet_column_indices_by_name<R: ChunkReader + 'static>(
    reader: &SerializedFileReader<R>,
    names: &[&str],
) -> Option<Vec<usize>> {
    let columns = reader.metadata().file_metadata().schema_descr().columns();
    names
        .iter()
        .map(|name| columns.iter().position(|column| column.name() == *name))
        .collect()
}

pub(crate) fn parquet_row_group_count_with_store(
    path: &Path,
    file_cache: Arc<ParquetFileCache>,
    store: &dyn ObjectStore,
) -> Result<usize> {
    if file_cache.enabled() {
        let reader = CachedParquetChunkReader::new(path, store, file_cache)?;
        let reader = SerializedFileReader::new(reader)?;
        return Ok(reader.metadata().num_row_groups());
    }
    let file = store.open(path)?;
    let reader = SerializedFileReader::new(file)?;
    Ok(reader.metadata().num_row_groups())
}

pub(crate) fn parquet_total_row_count_with_store(
    path: &Path,
    file_cache: Arc<ParquetFileCache>,
    store: &dyn ObjectStore,
) -> Result<usize> {
    if file_cache.enabled() {
        let reader = CachedParquetChunkReader::new(path, store, file_cache)?;
        let reader = SerializedFileReader::new(reader)?;
        return usize::try_from(reader.metadata().file_metadata().num_rows()).map_err(|_| {
            DodamError::UnsupportedSql("Parquet row count does not fit usize".to_string())
        });
    }
    let file = store.open(path)?;
    let reader = SerializedFileReader::new(file)?;
    usize::try_from(reader.metadata().file_metadata().num_rows())
        .map_err(|_| DodamError::UnsupportedSql("Parquet row count does not fit usize".to_string()))
}

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct DirectColumnScanMetrics {
    pub row_groups: usize,
    pub batches: usize,
    pub rows: usize,
    pub read_nanos: u64,
    pub consume_nanos: u64,
    pub selected_rows: usize,
    pub selected_runs: usize,
    pub selected_predicate_nanos: u64,
    pub selected_payload_nanos: u64,
}

impl DirectColumnScanMetrics {
    fn add_read_nanos(&mut self, nanos: u64) {
        self.read_nanos = self.read_nanos.saturating_add(nanos);
    }

    fn add_consume_nanos(&mut self, nanos: u64) {
        self.consume_nanos = self.consume_nanos.saturating_add(nanos);
    }

    fn add_selected_predicate_nanos(&mut self, nanos: u64) {
        self.selected_predicate_nanos = self.selected_predicate_nanos.saturating_add(nanos);
    }

    fn add_selected_payload_nanos(&mut self, nanos: u64) {
        self.selected_payload_nanos = self.selected_payload_nanos.saturating_add(nanos);
    }
}

pub(crate) struct DirectByteArrayPayloadReader<'a> {
    reader: &'a mut ColumnReaderImpl<ByteArrayType>,
    read_nanos: u64,
}

impl<'a> DirectByteArrayPayloadReader<'a> {
    fn new(reader: &'a mut ColumnReaderImpl<ByteArrayType>) -> Self {
        Self {
            reader,
            read_nanos: 0,
        }
    }

    pub(crate) fn read_records(
        &mut self,
        records: usize,
        def_levels: &mut Vec<i16>,
        values: &mut Vec<ByteArray>,
    ) -> Result<(usize, usize, usize)> {
        let started = Instant::now();
        let result = self
            .reader
            .read_records(records, Some(def_levels), None, values)?;
        self.read_nanos = self.read_nanos.saturating_add(elapsed_nanos(started));
        Ok(result)
    }

    pub(crate) fn skip_records(&mut self, records: usize) -> Result<usize> {
        let started = Instant::now();
        let skipped = self.reader.skip_records(records)?;
        self.read_nanos = self.read_nanos.saturating_add(elapsed_nanos(started));
        Ok(skipped)
    }

    fn take_read_nanos(&mut self) -> u64 {
        let read_nanos = self.read_nanos;
        self.read_nanos = 0;
        read_nanos
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn scan_parquet_i64_byte_array_payload_columns_with_store<F>(
    path: &Path,
    batch_size: usize,
    row_groups: &[usize],
    columns: [&str; 2],
    file_cache: Arc<ParquetFileCache>,
    store: &dyn ObjectStore,
    consume: F,
) -> Result<Option<DirectColumnScanMetrics>>
where
    F: for<'a> FnMut(&[i64], &mut DirectByteArrayPayloadReader<'a>) -> Result<Option<()>>,
{
    if file_cache.enabled() {
        let reader = CachedParquetChunkReader::new(path, store, file_cache)?;
        let reader = SerializedFileReader::new(reader)?;
        return scan_parquet_i64_byte_array_payload_columns_reader(
            reader, batch_size, row_groups, columns, consume,
        );
    }
    let file = store.open(path)?;
    let reader = SerializedFileReader::new(file)?;
    scan_parquet_i64_byte_array_payload_columns_reader(
        reader, batch_size, row_groups, columns, consume,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn scan_parquet_i32_i64_byte_array_columns_with_store<F>(
    path: &Path,
    batch_size: usize,
    row_groups: &[usize],
    columns: [&str; 3],
    file_cache: Arc<ParquetFileCache>,
    store: &dyn ObjectStore,
    consume: F,
) -> Result<Option<DirectColumnScanMetrics>>
where
    F: FnMut(&[i32], &[i64], &[i16], &[ByteArray]) -> Result<Option<()>>,
{
    if file_cache.enabled() {
        let reader = CachedParquetChunkReader::new(path, store, file_cache)?;
        let reader = SerializedFileReader::new(reader)?;
        return scan_parquet_i32_i64_byte_array_columns_reader(
            reader, batch_size, row_groups, columns, consume,
        );
    }
    let file = store.open(path)?;
    let reader = SerializedFileReader::new(file)?;
    scan_parquet_i32_i64_byte_array_columns_reader(reader, batch_size, row_groups, columns, consume)
}

pub(crate) fn scan_parquet_i32_byte_array_columns_with_store<F>(
    path: &Path,
    batch_size: usize,
    row_groups: &[usize],
    columns: [&str; 2],
    file_cache: Arc<ParquetFileCache>,
    store: &dyn ObjectStore,
    consume: F,
) -> Result<Option<DirectColumnScanMetrics>>
where
    F: FnMut(&[i32], &[i16], &[ByteArray]) -> Result<Option<()>>,
{
    if file_cache.enabled() {
        let reader = CachedParquetChunkReader::new(path, store, file_cache)?;
        let reader = SerializedFileReader::new(reader)?;
        return scan_parquet_i32_byte_array_columns_reader(
            reader, batch_size, row_groups, columns, consume,
        );
    }
    let file = store.open(path)?;
    let reader = SerializedFileReader::new(file)?;
    scan_parquet_i32_byte_array_columns_reader(reader, batch_size, row_groups, columns, consume)
}

pub(crate) fn scan_parquet_i32_i32_columns_with_store<F>(
    path: &Path,
    batch_size: usize,
    row_groups: &[usize],
    columns: [&str; 2],
    file_cache: Arc<ParquetFileCache>,
    store: &dyn ObjectStore,
    consume: F,
) -> Result<Option<DirectColumnScanMetrics>>
where
    F: FnMut(&[i32], Option<&[i16]>, &[i32], Option<&[i16]>) -> Result<Option<()>>,
{
    if file_cache.enabled() {
        let reader = CachedParquetChunkReader::new(path, store, file_cache)?;
        let reader = SerializedFileReader::new(reader)?;
        return scan_parquet_i32_i32_columns_reader(
            reader, batch_size, row_groups, columns, consume,
        );
    }
    let file = store.open(path)?;
    let reader = SerializedFileReader::new(file)?;
    scan_parquet_i32_i32_columns_reader(reader, batch_size, row_groups, columns, consume)
}

pub(crate) fn scan_parquet_i32_byte_array_selected_by_i32_with_store<P, F>(
    path: &Path,
    row_groups: &[usize],
    columns: [&str; 2],
    file_cache: Arc<ParquetFileCache>,
    store: &dyn ObjectStore,
    predicate: P,
    consume: F,
) -> Result<Option<DirectColumnScanMetrics>>
where
    P: Fn(i32) -> bool,
    F: FnMut(&[i32], &[i32], &[Bytes]) -> Result<Option<()>>,
{
    if file_cache.enabled() {
        let reader = CachedParquetChunkReader::new(path, store, file_cache)?;
        let reader = SerializedFileReader::new(reader)?;
        return scan_parquet_i32_byte_array_selected_by_i32_reader(
            reader, row_groups, columns, predicate, consume,
        );
    }
    let file = store.open(path)?;
    let reader = SerializedFileReader::new(file)?;
    scan_parquet_i32_byte_array_selected_by_i32_reader(
        reader, row_groups, columns, predicate, consume,
    )
}

pub(crate) fn scan_parquet_i32_selected_by_byte_array_prefix_with_store<F>(
    path: &Path,
    row_groups: &[usize],
    columns: [&str; 2],
    prefix: &[u8],
    file_cache: Arc<ParquetFileCache>,
    store: &dyn ObjectStore,
    consume: F,
) -> Result<Option<DirectColumnScanMetrics>>
where
    F: FnMut(&[i32], &[i32], &[Bytes]) -> Result<Option<()>>,
{
    if file_cache.enabled() {
        let reader = CachedParquetChunkReader::new(path, store, file_cache)?;
        let reader = SerializedFileReader::new(reader)?;
        return scan_parquet_i32_selected_by_byte_array_prefix_reader(
            reader, row_groups, columns, prefix, consume,
        );
    }
    let file = store.open(path)?;
    let reader = SerializedFileReader::new(file)?;
    scan_parquet_i32_selected_by_byte_array_prefix_reader(
        reader, row_groups, columns, prefix, consume,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn scan_parquet_i32_i64_dictionary_id_columns_with_store<F>(
    path: &Path,
    batch_size: usize,
    row_groups: &[usize],
    columns: [&str; 3],
    file_cache: Arc<ParquetFileCache>,
    store: &dyn ObjectStore,
    consume: F,
) -> Result<Option<DirectColumnScanMetrics>>
where
    F: FnMut(&[i32], &[i64], Option<&[i16]>, &[i32], &[Bytes]) -> Result<Option<()>>,
{
    if file_cache.enabled() {
        let reader = CachedParquetChunkReader::new(path, store, file_cache)?;
        let reader = SerializedFileReader::new(reader)?;
        return scan_parquet_i32_i64_dictionary_id_columns_reader(
            reader, batch_size, row_groups, columns, consume,
        );
    }
    let file = store.open(path)?;
    let reader = SerializedFileReader::new(file)?;
    scan_parquet_i32_i64_dictionary_id_columns_reader(
        reader, batch_size, row_groups, columns, consume,
    )
}

pub(crate) fn scan_parquet_i32_dictionary_id_columns_with_store<F>(
    path: &Path,
    batch_size: usize,
    row_groups: &[usize],
    columns: [&str; 2],
    file_cache: Arc<ParquetFileCache>,
    store: &dyn ObjectStore,
    consume: F,
) -> Result<Option<DirectColumnScanMetrics>>
where
    F: FnMut(&[i32], Option<&[i16]>, &[i32], &[Bytes]) -> Result<Option<()>>,
{
    if file_cache.enabled() {
        let reader = CachedParquetChunkReader::new(path, store, file_cache)?;
        let reader = SerializedFileReader::new(reader)?;
        return scan_parquet_i32_dictionary_id_columns_reader(
            reader, batch_size, row_groups, columns, consume,
        );
    }
    let file = store.open(path)?;
    let reader = SerializedFileReader::new(file)?;
    scan_parquet_i32_dictionary_id_columns_reader(reader, batch_size, row_groups, columns, consume)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn scan_parquet_i64_dictionary_i32x3_columns_with_store<F>(
    path: &Path,
    batch_size: usize,
    row_groups: &[usize],
    columns: [&str; 5],
    file_cache: Arc<ParquetFileCache>,
    store: &dyn ObjectStore,
    consume: F,
) -> Result<Option<DirectColumnScanMetrics>>
where
    F: FnMut(&[i64], &[i32], &[Bytes], &[i32], &[i32], &[i32]) -> Result<Option<()>>,
{
    if file_cache.enabled() {
        let reader = CachedParquetChunkReader::new(path, store, file_cache)?;
        let reader = SerializedFileReader::new(reader)?;
        return scan_parquet_i64_dictionary_i32x3_columns_reader(
            reader, batch_size, row_groups, columns, consume,
        );
    }
    let file = store.open(path)?;
    let reader = SerializedFileReader::new(file)?;
    scan_parquet_i64_dictionary_i32x3_columns_reader(
        reader, batch_size, row_groups, columns, consume,
    )
}

pub(crate) fn scan_parquet_i64_dictionary_i32x3_page_columns_with_store<F>(
    path: &Path,
    batch_size: usize,
    row_groups: &[usize],
    columns: [&str; 5],
    file_cache: Arc<ParquetFileCache>,
    store: &dyn ObjectStore,
    consume: F,
) -> Result<Option<DirectColumnScanMetrics>>
where
    F: FnMut(&[u8], &[i32], &[Bytes], &[u8], &[u8], &[u8], usize) -> Result<Option<()>>,
{
    if file_cache.enabled() {
        let reader = CachedParquetChunkReader::new(path, store, file_cache)?;
        let reader = SerializedFileReader::new(reader)?;
        return scan_parquet_i64_dictionary_i32x3_page_columns_reader(
            reader, batch_size, row_groups, columns, consume,
        );
    }
    let file = store.open(path)?;
    let reader = SerializedFileReader::new(file)?;
    scan_parquet_i64_dictionary_i32x3_page_columns_reader(
        reader, batch_size, row_groups, columns, consume,
    )
}

pub(crate) fn scan_parquet_i64_dictionary_i32x3_dict_columns_with_store<F>(
    path: &Path,
    batch_size: usize,
    row_groups: &[usize],
    columns: [&str; 5],
    file_cache: Arc<ParquetFileCache>,
    store: &dyn ObjectStore,
    consume: F,
) -> Result<Option<DirectColumnScanMetrics>>
where
    F: FnMut(
        &[i32],
        &[i64],
        &[i32],
        &[Bytes],
        &[i32],
        &[i32],
        &[i32],
        &[i32],
        &[i32],
        &[i32],
    ) -> Result<Option<()>>,
{
    if file_cache.enabled() {
        let reader = CachedParquetChunkReader::new(path, store, file_cache)?;
        let reader = SerializedFileReader::new(reader)?;
        return scan_parquet_i64_dictionary_i32x3_dict_columns_reader(
            reader, batch_size, row_groups, columns, consume,
        );
    }
    let file = store.open(path)?;
    let reader = SerializedFileReader::new(file)?;
    scan_parquet_i64_dictionary_i32x3_dict_columns_reader(
        reader, batch_size, row_groups, columns, consume,
    )
}

pub(crate) fn scan_parquet_i64_byte_array_selected_by_i32x3_dictionary_with_store<P, F>(
    path: &Path,
    row_groups: &[usize],
    columns: [&str; 5],
    file_cache: Arc<ParquetFileCache>,
    store: &dyn ObjectStore,
    predicate: P,
    consume: F,
) -> Result<Option<DirectColumnScanMetrics>>
where
    P: Fn(i32, i32, i32) -> bool,
    F: FnMut(&[i64], &[i32], &[Bytes]) -> Result<Option<()>>,
{
    if file_cache.enabled() {
        let reader = CachedParquetChunkReader::new(path, store, file_cache)?;
        let reader = SerializedFileReader::new(reader)?;
        return scan_parquet_i64_byte_array_selected_by_i32x3_dictionary_reader(
            reader, row_groups, columns, predicate, consume,
        );
    }
    let file = store.open(path)?;
    let reader = SerializedFileReader::new(file)?;
    scan_parquet_i64_byte_array_selected_by_i32x3_dictionary_reader(
        reader, row_groups, columns, predicate, consume,
    )
}

#[allow(clippy::too_many_arguments)]
fn scan_parquet_i64_byte_array_selected_by_i32x3_dictionary_reader<R, P, F>(
    reader: SerializedFileReader<R>,
    row_groups: &[usize],
    columns: [&str; 5],
    predicate: P,
    mut consume: F,
) -> Result<Option<DirectColumnScanMetrics>>
where
    R: ChunkReader + 'static,
    P: Fn(i32, i32, i32) -> bool,
    F: FnMut(&[i64], &[i32], &[Bytes]) -> Result<Option<()>>,
{
    let trace_fallback = || {
        std::env::var_os("DODAM_TPCH_PROFILE").is_some()
            || std::env::var_os("DODAM_DIRECT_PRIMITIVE_PROFILE").is_some()
    };
    let Some(column_indices) = parquet_column_indices_by_name(&reader, &columns) else {
        if trace_fallback() {
            eprintln!("[dodam:direct-selected-i32x3-fallback] missing columns");
        }
        return Ok(None);
    };
    let [
        key_column,
        byte_array_column,
        first_column,
        second_column,
        third_column,
    ] = <[usize; 5]>::try_from(column_indices).map_err(|_| {
        DodamError::UnsupportedSql("direct parquet column index shape mismatch".to_string())
    })?;
    let schema = reader.metadata().file_metadata().schema_descr();
    let key_required = schema.column(key_column).max_def_level() == 0;
    let mut metrics = DirectColumnScanMetrics {
        row_groups: row_groups.len(),
        ..DirectColumnScanMetrics::default()
    };
    for &row_group_index in row_groups {
        let row_group = reader.get_row_group(row_group_index)?;
        let row_count = usize::try_from(row_group.metadata().num_rows()).map_err(|_| {
            DodamError::UnsupportedSql("row group row count out of range".to_string())
        })?;

        let read_started = Instant::now();
        let predicate_started = Instant::now();
        let mut selected_runs =
            Vec::<(usize, usize)>::with_capacity((row_count / 64).clamp(16, 16_384));
        let mut selected_builder = SelectionRunsBuilder::default();
        if build_i32x3_dictionary_selected_runs_for_row_group_pagewise(
            &*row_group,
            [first_column, second_column, third_column],
            row_count,
            &predicate,
            &mut selected_runs,
            &mut selected_builder,
        )?
        .is_none()
        {
            let Some((first_ids, first_dictionary)) =
                read_i32_dictionary_ids_for_row_group(&*row_group, first_column)?
            else {
                if trace_fallback() {
                    eprintln!(
                        "[dodam:direct-selected-i32x3-fallback] first predicate dictionary unsupported"
                    );
                }
                return Ok(None);
            };
            let Some((second_ids, second_dictionary)) =
                read_i32_dictionary_ids_for_row_group(&*row_group, second_column)?
            else {
                if trace_fallback() {
                    eprintln!(
                        "[dodam:direct-selected-i32x3-fallback] second predicate dictionary unsupported"
                    );
                }
                return Ok(None);
            };
            let Some((third_ids, third_dictionary)) =
                read_i32_dictionary_ids_for_row_group(&*row_group, third_column)?
            else {
                if trace_fallback() {
                    eprintln!(
                        "[dodam:direct-selected-i32x3-fallback] third predicate dictionary unsupported"
                    );
                }
                return Ok(None);
            };
            if first_ids.len() != row_count
                || second_ids.len() != row_count
                || third_ids.len() != row_count
            {
                return Ok(None);
            }
            append_i32x3_dictionary_selected_runs(
                &first_ids,
                &first_dictionary,
                &second_ids,
                &second_dictionary,
                &third_ids,
                &third_dictionary,
                &predicate,
                &mut selected_runs,
                &mut selected_builder,
            );
        }
        metrics.add_selected_predicate_nanos(elapsed_nanos(predicate_started));
        metrics.rows = metrics.rows.saturating_add(row_count);
        metrics.add_read_nanos(elapsed_nanos(read_started));

        let selected_rows = selected_builder.selected_rows();
        metrics.selected_rows = metrics.selected_rows.saturating_add(selected_rows);
        metrics.selected_runs = metrics.selected_runs.saturating_add(selected_runs.len());
        if selected_rows == 0 {
            continue;
        }

        let payload_started = Instant::now();
        let fragmented_payload =
            fragmented_selected_payload_should_full_decode(selected_rows, selected_runs.len());
        let full_payload = if fragmented_payload {
            if let Some((key_ids, key_dictionary)) =
                read_i64_dictionary_ids_for_row_group(&*row_group, key_column)?
            {
                if let Some((mode_def_levels, mode_ids, mode_dictionary)) =
                    read_byte_array_dictionary_ids_for_row_group(&*row_group, byte_array_column)?
                {
                    if mode_def_levels.is_empty()
                        && key_ids.len() == row_count
                        && mode_ids.len() == row_count
                    {
                        let selected_offsets =
                            materialize_selected_u32_offsets(&selected_runs, selected_rows)?;
                        let mut keys = Vec::<i64>::with_capacity(selected_rows);
                        compact_i64_dictionary_selected_offsets(
                            &key_ids,
                            &key_dictionary,
                            &selected_offsets,
                            &mut keys,
                        )?;
                        let mut selected_mode_ids = Vec::<i32>::with_capacity(selected_rows);
                        compact_selected_i32_offsets(
                            &mode_ids,
                            &selected_offsets,
                            &mut selected_mode_ids,
                        )?;
                        Some((keys, selected_mode_ids, mode_dictionary))
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };
        let (selected_keys, selected_byte_array_ids, byte_array_dictionary) = if let Some(payload) =
            full_payload
        {
            payload
        } else {
            let selected_keys = if let Some(keys) =
                read_i64_dictionary_values_selected_for_row_group(
                    &*row_group,
                    key_column,
                    &selected_runs,
                )? {
                keys
            } else if let Some((key_ids, key_dictionary)) =
                read_i64_dictionary_ids_for_row_group(&*row_group, key_column)?
            {
                let mut keys = Vec::<i64>::with_capacity(selected_rows);
                compact_i64_dictionary_selected_runs(
                    &key_ids,
                    &key_dictionary,
                    &selected_runs,
                    &mut keys,
                )?;
                keys
            } else {
                let mut keys = Vec::<i64>::with_capacity(selected_rows);
                let mut key_def_levels = Vec::<i16>::new();
                let mut key_metrics = DirectPrimitiveColumnScanMetrics::default();
                let mut key_reader = match row_group.get_column_reader(key_column)? {
                    ColumnReader::Int64ColumnReader(reader) => reader,
                    _ => {
                        if trace_fallback() {
                            eprintln!(
                                "[dodam:direct-selected-i32x3-fallback] key column is not int64"
                            );
                        }
                        return Ok(None);
                    }
                };
                if !read_i64_selected_runs(
                    &mut key_reader,
                    row_count,
                    &selected_runs,
                    key_required,
                    &mut key_def_levels,
                    &mut keys,
                    &mut key_metrics,
                )? {
                    if trace_fallback() {
                        eprintln!(
                            "[dodam:direct-selected-i32x3-fallback] selected key read failed"
                        );
                    }
                    return Ok(None);
                }
                keys
            };
            let Some((selected_byte_array_ids, byte_array_dictionary)) =
                read_byte_array_dictionary_ids_selected_for_row_group(
                    &*row_group,
                    byte_array_column,
                    &selected_runs,
                    &[],
                )?
            else {
                if trace_fallback() {
                    eprintln!(
                        "[dodam:direct-selected-i32x3-fallback] selected byte-array dictionary read failed"
                    );
                }
                return Ok(None);
            };
            (
                selected_keys,
                selected_byte_array_ids,
                byte_array_dictionary,
            )
        };
        if selected_keys.len() != selected_rows || selected_byte_array_ids.len() != selected_rows {
            if trace_fallback() {
                eprintln!(
                    "[dodam:direct-selected-i32x3-fallback] selected payload length mismatch keys={} ids={} selected={}",
                    selected_keys.len(),
                    selected_byte_array_ids.len(),
                    selected_rows
                );
            }
            return Ok(None);
        }
        let payload_nanos = elapsed_nanos(payload_started);
        metrics.add_selected_payload_nanos(payload_nanos);
        metrics.add_read_nanos(payload_nanos);
        metrics.batches += 1;

        let consume_started = Instant::now();
        if consume(
            &selected_keys,
            &selected_byte_array_ids,
            &byte_array_dictionary,
        )?
        .is_none()
        {
            return Ok(None);
        }
        metrics.add_consume_nanos(elapsed_nanos(consume_started));
    }
    Ok(Some(metrics))
}

#[allow(clippy::too_many_arguments)]
fn append_i32x3_dictionary_selected_runs<P>(
    first_ids: &[i32],
    first_dictionary: &[i32],
    second_ids: &[i32],
    second_dictionary: &[i32],
    third_ids: &[i32],
    third_dictionary: &[i32],
    predicate: &P,
    runs: &mut Vec<(usize, usize)>,
    builder: &mut SelectionRunsBuilder,
) where
    P: Fn(i32, i32, i32) -> bool,
{
    let rows = first_ids.len().min(second_ids.len()).min(third_ids.len());
    let mut run_start = None::<usize>;
    let mut run_len = 0usize;
    for row in 0..rows {
        let selected = i32_dictionary_lookup(first_ids[row], first_dictionary)
            .zip(i32_dictionary_lookup(second_ids[row], second_dictionary))
            .zip(i32_dictionary_lookup(third_ids[row], third_dictionary))
            .is_some_and(|((first, second), third)| predicate(first, second, third));
        if selected {
            if run_start.is_none() {
                run_start = Some(row);
                run_len = 1;
            } else {
                run_len += 1;
            }
        } else if let Some(start) = run_start.take() {
            builder.push_disjoint_run(runs, start, run_len);
            run_len = 0;
        }
    }
    if let Some(start) = run_start {
        builder.push_disjoint_run(runs, start, run_len);
    }
}

fn compact_i64_dictionary_selected_runs(
    ids: &[i32],
    dictionary: &[i64],
    runs: &[(usize, usize)],
    output: &mut Vec<i64>,
) -> Result<()> {
    for &(start, len) in runs {
        let end = start.saturating_add(len);
        let Some(slice) = ids.get(start..end) else {
            return Err(DodamError::UnsupportedSql(
                "selected i64 dictionary run is out of range".to_string(),
            ));
        };
        for &id in slice {
            let id = usize::try_from(id).map_err(|_| {
                DodamError::UnsupportedSql("negative i64 dictionary id".to_string())
            })?;
            let Some(value) = dictionary.get(id) else {
                return Err(DodamError::UnsupportedSql(
                    "i64 dictionary id is out of range".to_string(),
                ));
            };
            output.push(*value);
        }
    }
    Ok(())
}

fn materialize_selected_u32_offsets(
    runs: &[(usize, usize)],
    selected_rows: usize,
) -> Result<Vec<u32>> {
    let mut offsets = Vec::with_capacity(selected_rows);
    for &(start, len) in runs {
        let end = start.checked_add(len).ok_or_else(|| {
            DodamError::UnsupportedSql("selected run end is out of range".to_string())
        })?;
        if end > usize::try_from(u32::MAX).unwrap_or(usize::MAX) {
            return Err(DodamError::UnsupportedSql(
                "selected offset is out of u32 range".to_string(),
            ));
        }
        offsets.extend((start..end).map(|offset| offset as u32));
    }
    if offsets.len() != selected_rows {
        return Err(DodamError::UnsupportedSql(
            "selected offset count mismatch".to_string(),
        ));
    }
    Ok(offsets)
}

fn compact_i64_dictionary_selected_offsets(
    ids: &[i32],
    dictionary: &[i64],
    offsets: &[u32],
    output: &mut Vec<i64>,
) -> Result<()> {
    for &offset in offsets {
        let Some(&id) = ids.get(offset as usize) else {
            return Err(DodamError::UnsupportedSql(
                "selected i64 dictionary offset is out of range".to_string(),
            ));
        };
        let id = usize::try_from(id)
            .map_err(|_| DodamError::UnsupportedSql("negative i64 dictionary id".to_string()))?;
        let Some(value) = dictionary.get(id) else {
            return Err(DodamError::UnsupportedSql(
                "i64 dictionary id is out of range".to_string(),
            ));
        };
        output.push(*value);
    }
    Ok(())
}

fn compact_selected_i32_offsets(
    values: &[i32],
    offsets: &[u32],
    output: &mut Vec<i32>,
) -> Result<()> {
    for &offset in offsets {
        let Some(&value) = values.get(offset as usize) else {
            return Err(DodamError::UnsupportedSql(
                "selected i32 offset is out of range".to_string(),
            ));
        };
        output.push(value);
    }
    Ok(())
}

fn fragmented_selected_payload_should_full_decode(
    selected_rows: usize,
    selected_runs: usize,
) -> bool {
    let min_selected_runs = std::env::var("DODAM_FRAGMENTED_FULL_PAYLOAD_MIN_RUNS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(4096);
    let max_average_run_len = std::env::var("DODAM_FRAGMENTED_FULL_PAYLOAD_MAX_AVG_RUN_LEN")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(2);
    choose_fragmented_selected_payload_full_decode(FragmentedSelectedPayloadCostInput {
        selected_rows,
        selected_runs,
        min_selected_runs,
        max_average_run_len,
    })
}

#[inline(always)]
fn i32_dictionary_lookup(id: i32, dictionary: &[i32]) -> Option<i32> {
    let id = usize::try_from(id).ok()?;
    dictionary.get(id).copied()
}

#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn scan_parquet_i64_dictionary_i32x3_dict_columns_reader<R, F>(
    reader: SerializedFileReader<R>,
    batch_size: usize,
    row_groups: &[usize],
    columns: [&str; 5],
    mut consume: F,
) -> Result<Option<DirectColumnScanMetrics>>
where
    R: ChunkReader + 'static,
    F: FnMut(
        &[i32],
        &[i64],
        &[i32],
        &[Bytes],
        &[i32],
        &[i32],
        &[i32],
        &[i32],
        &[i32],
        &[i32],
    ) -> Result<Option<()>>,
{
    let Some(column_indices) = parquet_column_indices_by_name(&reader, &columns) else {
        return Ok(None);
    };
    let [
        key_column,
        dictionary_column,
        first_column,
        second_column,
        third_column,
    ] = <[usize; 5]>::try_from(column_indices).map_err(|_| {
        DodamError::UnsupportedSql("direct parquet column index shape mismatch".to_string())
    })?;
    let mut metrics = DirectColumnScanMetrics {
        row_groups: row_groups.len(),
        ..DirectColumnScanMetrics::default()
    };
    for &row_group_index in row_groups {
        let row_group = reader.get_row_group(row_group_index)?;
        let row_count = usize::try_from(row_group.metadata().num_rows()).map_err(|_| {
            DodamError::UnsupportedSql("row group row count out of range".to_string())
        })?;
        let read_started = Instant::now();
        let Some((key_ids, key_dictionary)) =
            read_i64_dictionary_ids_for_row_group(&*row_group, key_column)?
        else {
            return Ok(None);
        };
        let Some((mode_def_levels, mode_ids, mode_dictionary)) =
            read_byte_array_dictionary_ids_for_row_group(&*row_group, dictionary_column)?
        else {
            return Ok(None);
        };
        if !mode_def_levels.is_empty() && !direct_all_present(false, &mode_def_levels) {
            return Ok(None);
        }
        let Some((first_ids, first_dictionary)) =
            read_i32_dictionary_ids_for_row_group(&*row_group, first_column)?
        else {
            return Ok(None);
        };
        let Some((second_ids, second_dictionary)) =
            read_i32_dictionary_ids_for_row_group(&*row_group, second_column)?
        else {
            return Ok(None);
        };
        let Some((third_ids, third_dictionary)) =
            read_i32_dictionary_ids_for_row_group(&*row_group, third_column)?
        else {
            return Ok(None);
        };
        if key_ids.len() != row_count
            || mode_ids.len() != row_count
            || first_ids.len() != row_count
            || second_ids.len() != row_count
            || third_ids.len() != row_count
        {
            return Ok(None);
        }
        metrics.add_read_nanos(elapsed_nanos(read_started));
        let mut row_offset = 0usize;
        while row_offset < row_count {
            let records = batch_size.min(row_count - row_offset);
            metrics.batches += 1;
            metrics.rows = metrics.rows.saturating_add(records);
            let consume_started = Instant::now();
            if consume(
                &key_ids[row_offset..row_offset + records],
                &key_dictionary,
                &mode_ids[row_offset..row_offset + records],
                &mode_dictionary,
                &first_ids[row_offset..row_offset + records],
                &first_dictionary,
                &second_ids[row_offset..row_offset + records],
                &second_dictionary,
                &third_ids[row_offset..row_offset + records],
                &third_dictionary,
            )?
            .is_none()
            {
                return Ok(None);
            }
            metrics.add_consume_nanos(elapsed_nanos(consume_started));
            row_offset += records;
        }
    }
    Ok(Some(metrics))
}

#[allow(clippy::too_many_arguments)]
fn scan_parquet_i64_dictionary_i32x3_page_columns_reader<R, F>(
    reader: SerializedFileReader<R>,
    batch_size: usize,
    row_groups: &[usize],
    columns: [&str; 5],
    mut consume: F,
) -> Result<Option<DirectColumnScanMetrics>>
where
    R: ChunkReader + 'static,
    F: FnMut(&[u8], &[i32], &[Bytes], &[u8], &[u8], &[u8], usize) -> Result<Option<()>>,
{
    let Some(column_indices) = parquet_column_indices_by_name(&reader, &columns) else {
        return Ok(None);
    };
    let [
        key_column,
        dictionary_column,
        first_column,
        second_column,
        third_column,
    ] = <[usize; 5]>::try_from(column_indices).map_err(|_| {
        DodamError::UnsupportedSql("direct parquet column index shape mismatch".to_string())
    })?;
    let schema = reader.metadata().file_metadata().schema_descr();
    if schema.column(key_column).physical_type() != ParquetPhysicalType::INT64
        || schema.column(dictionary_column).physical_type() != ParquetPhysicalType::BYTE_ARRAY
        || schema.column(first_column).physical_type() != ParquetPhysicalType::INT32
        || schema.column(second_column).physical_type() != ParquetPhysicalType::INT32
        || schema.column(third_column).physical_type() != ParquetPhysicalType::INT32
        || schema.column(key_column).max_rep_level() != 0
        || schema.column(dictionary_column).max_rep_level() != 0
        || schema.column(first_column).max_rep_level() != 0
        || schema.column(second_column).max_rep_level() != 0
        || schema.column(third_column).max_rep_level() != 0
    {
        return Ok(None);
    }
    let mut metrics = DirectColumnScanMetrics {
        row_groups: row_groups.len(),
        ..DirectColumnScanMetrics::default()
    };
    for &row_group_index in row_groups {
        let row_group = reader.get_row_group(row_group_index)?;
        let row_count = usize::try_from(row_group.metadata().num_rows()).map_err(|_| {
            DodamError::UnsupportedSql("row group row count out of range".to_string())
        })?;
        let read_started = Instant::now();
        let Some((dictionary_def_levels, dictionary_ids, dictionary)) =
            read_byte_array_dictionary_ids_for_row_group(&*row_group, dictionary_column)?
        else {
            return Ok(None);
        };
        if !dictionary_def_levels.is_empty() && !direct_all_present(false, &dictionary_def_levels) {
            return Ok(None);
        }
        if dictionary_ids.len() != row_count {
            return Ok(None);
        }
        metrics.add_read_nanos(elapsed_nanos(read_started));

        let mut key_cursor = RequiredPlainPrimitivePageCursor::new(
            row_group.get_column_page_reader(key_column)?,
            DirectPrimitiveColumnType::I64,
            schema.column(key_column).max_def_level(),
        );
        let mut first_cursor = RequiredPlainPrimitivePageCursor::new(
            row_group.get_column_page_reader(first_column)?,
            DirectPrimitiveColumnType::I32,
            schema.column(first_column).max_def_level(),
        );
        let mut second_cursor = RequiredPlainPrimitivePageCursor::new(
            row_group.get_column_page_reader(second_column)?,
            DirectPrimitiveColumnType::I32,
            schema.column(second_column).max_def_level(),
        );
        let mut third_cursor = RequiredPlainPrimitivePageCursor::new(
            row_group.get_column_page_reader(third_column)?,
            DirectPrimitiveColumnType::I32,
            schema.column(third_column).max_def_level(),
        );
        let mut rows_read = 0usize;
        while rows_read < row_count {
            let read_started = Instant::now();
            let loaded = key_cursor.ensure_page()?
                && first_cursor.ensure_page()?
                && second_cursor.ensure_page()?
                && third_cursor.ensure_page()?;
            if !loaded {
                metrics.add_read_nanos(elapsed_nanos(read_started));
                return Ok(None);
            }
            let records = batch_size
                .min(row_count - rows_read)
                .min(key_cursor.available_rows())
                .min(first_cursor.available_rows())
                .min(second_cursor.available_rows())
                .min(third_cursor.available_rows());
            let Some(key_bytes) = key_cursor.raw_bytes(records) else {
                return Ok(None);
            };
            let Some(first_bytes) = first_cursor.raw_bytes(records) else {
                return Ok(None);
            };
            let Some(second_bytes) = second_cursor.raw_bytes(records) else {
                return Ok(None);
            };
            let Some(third_bytes) = third_cursor.raw_bytes(records) else {
                return Ok(None);
            };
            let mode_ids = &dictionary_ids[rows_read..rows_read + records];
            metrics.add_read_nanos(elapsed_nanos(read_started));
            metrics.batches += 1;
            metrics.rows = metrics.rows.saturating_add(records);
            let consume_started = Instant::now();
            if consume(
                key_bytes,
                mode_ids,
                &dictionary,
                first_bytes,
                second_bytes,
                third_bytes,
                records,
            )?
            .is_none()
            {
                return Ok(None);
            }
            metrics.add_consume_nanos(elapsed_nanos(consume_started));
            key_cursor.advance(records);
            first_cursor.advance(records);
            second_cursor.advance(records);
            third_cursor.advance(records);
            rows_read += records;
        }
    }
    Ok(Some(metrics))
}

#[allow(clippy::too_many_arguments)]
fn scan_parquet_i64_dictionary_i32x3_columns_reader<R, F>(
    reader: SerializedFileReader<R>,
    batch_size: usize,
    row_groups: &[usize],
    columns: [&str; 5],
    mut consume: F,
) -> Result<Option<DirectColumnScanMetrics>>
where
    R: ChunkReader + 'static,
    F: FnMut(&[i64], &[i32], &[Bytes], &[i32], &[i32], &[i32]) -> Result<Option<()>>,
{
    let Some(column_indices) = parquet_column_indices_by_name(&reader, &columns) else {
        return Ok(None);
    };
    let [
        key_column,
        dictionary_column,
        first_column,
        second_column,
        third_column,
    ] = <[usize; 5]>::try_from(column_indices).map_err(|_| {
        DodamError::UnsupportedSql("direct parquet column index shape mismatch".to_string())
    })?;
    let schema = reader.metadata().file_metadata().schema_descr();
    let key_required = schema.column(key_column).max_def_level() == 0;
    let first_required = schema.column(first_column).max_def_level() == 0;
    let second_required = schema.column(second_column).max_def_level() == 0;
    let third_required = schema.column(third_column).max_def_level() == 0;
    let mut metrics = DirectColumnScanMetrics {
        row_groups: row_groups.len(),
        ..DirectColumnScanMetrics::default()
    };
    for &row_group_index in row_groups {
        let row_group = reader.get_row_group(row_group_index)?;
        let read_started = Instant::now();
        let Some((dictionary_def_levels, dictionary_ids, dictionary)) =
            read_byte_array_dictionary_ids_for_row_group(&*row_group, dictionary_column)?
        else {
            return Ok(None);
        };
        if !dictionary_def_levels.is_empty() && !direct_all_present(false, &dictionary_def_levels) {
            return Ok(None);
        }
        metrics.add_read_nanos(elapsed_nanos(read_started));

        let mut key_reader = match row_group.get_column_reader(key_column)? {
            ColumnReader::Int64ColumnReader(reader) => reader,
            _ => return Ok(None),
        };
        let mut first_reader = match row_group.get_column_reader(first_column)? {
            ColumnReader::Int32ColumnReader(reader) => reader,
            _ => return Ok(None),
        };
        let mut second_reader = match row_group.get_column_reader(second_column)? {
            ColumnReader::Int32ColumnReader(reader) => reader,
            _ => return Ok(None),
        };
        let mut third_reader = match row_group.get_column_reader(third_column)? {
            ColumnReader::Int32ColumnReader(reader) => reader,
            _ => return Ok(None),
        };
        let mut key_values = Vec::<i64>::with_capacity(batch_size);
        let mut key_def_levels = Vec::<i16>::with_capacity(batch_size);
        let mut first_values = Vec::<i32>::with_capacity(batch_size);
        let mut first_def_levels = Vec::<i16>::with_capacity(batch_size);
        let mut second_values = Vec::<i32>::with_capacity(batch_size);
        let mut second_def_levels = Vec::<i16>::with_capacity(batch_size);
        let mut third_values = Vec::<i32>::with_capacity(batch_size);
        let mut third_def_levels = Vec::<i16>::with_capacity(batch_size);
        let mut row_offset = 0usize;
        loop {
            key_values.clear();
            key_def_levels.clear();
            first_values.clear();
            first_def_levels.clear();
            second_values.clear();
            second_def_levels.clear();
            third_values.clear();
            third_def_levels.clear();
            let read_started = Instant::now();
            let (records, key_value_count, _) = key_reader.read_records(
                batch_size,
                (!key_required).then_some(&mut key_def_levels),
                None,
                &mut key_values,
            )?;
            if records == 0 {
                metrics.add_read_nanos(elapsed_nanos(read_started));
                break;
            }
            let (first_records, first_value_count, _) = first_reader.read_records(
                records,
                (!first_required).then_some(&mut first_def_levels),
                None,
                &mut first_values,
            )?;
            let (second_records, second_value_count, _) = second_reader.read_records(
                records,
                (!second_required).then_some(&mut second_def_levels),
                None,
                &mut second_values,
            )?;
            let (third_records, third_value_count, _) = third_reader.read_records(
                records,
                (!third_required).then_some(&mut third_def_levels),
                None,
                &mut third_values,
            )?;
            metrics.add_read_nanos(elapsed_nanos(read_started));
            if first_records != records
                || second_records != records
                || third_records != records
                || key_value_count != records
                || first_value_count != records
                || second_value_count != records
                || third_value_count != records
                || (!key_required && !direct_all_present(false, &key_def_levels))
                || (!first_required && !direct_all_present(false, &first_def_levels))
                || (!second_required && !direct_all_present(false, &second_def_levels))
                || (!third_required && !direct_all_present(false, &third_def_levels))
                || row_offset + records > dictionary_ids.len()
            {
                return Ok(None);
            }
            let batch_dictionary_ids = &dictionary_ids[row_offset..row_offset + records];
            metrics.batches += 1;
            metrics.rows = metrics.rows.saturating_add(records);
            let consume_started = Instant::now();
            if consume(
                &key_values,
                batch_dictionary_ids,
                &dictionary,
                &first_values,
                &second_values,
                &third_values,
            )?
            .is_none()
            {
                return Ok(None);
            }
            metrics.add_consume_nanos(elapsed_nanos(consume_started));
            row_offset += records;
        }
        if row_offset != row_group.metadata().num_rows() as usize
            || row_offset != dictionary_ids.len()
        {
            return Ok(None);
        }
    }
    Ok(Some(metrics))
}

fn scan_parquet_i32_i64_dictionary_id_columns_reader<R, F>(
    reader: SerializedFileReader<R>,
    batch_size: usize,
    row_groups: &[usize],
    columns: [&str; 3],
    mut consume: F,
) -> Result<Option<DirectColumnScanMetrics>>
where
    R: ChunkReader + 'static,
    F: FnMut(&[i32], &[i64], Option<&[i16]>, &[i32], &[Bytes]) -> Result<Option<()>>,
{
    let Some(column_indices) = parquet_column_indices_by_name(&reader, &columns) else {
        return Ok(None);
    };
    let [predicate_column, sum_column, group_column] = <[usize; 3]>::try_from(column_indices)
        .map_err(|_| {
            DodamError::UnsupportedSql("direct parquet column index shape mismatch".to_string())
        })?;
    let schema = reader.metadata().file_metadata().schema_descr();
    let predicate_required = schema.column(predicate_column).max_def_level() == 0;
    let sum_required = schema.column(sum_column).max_def_level() == 0;
    let mut metrics = DirectColumnScanMetrics {
        row_groups: row_groups.len(),
        ..DirectColumnScanMetrics::default()
    };
    for &row_group_index in row_groups {
        let row_group = reader.get_row_group(row_group_index)?;
        let read_started = Instant::now();
        let Some((group_def_levels, group_ids, dictionary)) =
            read_byte_array_dictionary_ids_for_row_group(&*row_group, group_column)?
        else {
            return Ok(None);
        };
        metrics.add_read_nanos(elapsed_nanos(read_started));

        let mut predicate_reader = match row_group.get_column_reader(predicate_column)? {
            ColumnReader::Int32ColumnReader(reader) => reader,
            _ => return Ok(None),
        };
        let mut sum_reader = match row_group.get_column_reader(sum_column)? {
            ColumnReader::Int64ColumnReader(reader) => reader,
            _ => return Ok(None),
        };
        let mut predicate_values = Vec::<i32>::with_capacity(batch_size);
        let mut predicate_def_levels = Vec::<i16>::with_capacity(batch_size);
        let mut sum_values = Vec::<i64>::with_capacity(batch_size);
        let mut sum_def_levels = Vec::<i16>::with_capacity(batch_size);
        let mut row_offset = 0usize;
        let mut group_value_offset = 0usize;
        loop {
            predicate_values.clear();
            predicate_def_levels.clear();
            sum_values.clear();
            sum_def_levels.clear();
            let read_started = Instant::now();
            let (records, predicate_value_count, _) = predicate_reader.read_records(
                batch_size,
                (!predicate_required).then_some(&mut predicate_def_levels),
                None,
                &mut predicate_values,
            )?;
            if records == 0 {
                metrics.add_read_nanos(elapsed_nanos(read_started));
                break;
            }
            let (sum_records, sum_value_count, _) = sum_reader.read_records(
                records,
                (!sum_required).then_some(&mut sum_def_levels),
                None,
                &mut sum_values,
            )?;
            metrics.add_read_nanos(elapsed_nanos(read_started));
            if sum_records != records
                || predicate_value_count != records
                || sum_value_count != records
                || (!predicate_required && !direct_all_present(false, &predicate_def_levels))
                || (!sum_required && !direct_all_present(false, &sum_def_levels))
                || row_offset + records > row_group.metadata().num_rows() as usize
            {
                return Ok(None);
            }
            let batch_group_levels = if group_def_levels.is_empty() {
                None
            } else {
                Some(&group_def_levels[row_offset..row_offset + records])
            };
            let batch_group_values = match batch_group_levels {
                Some(levels) => levels.iter().filter(|level| **level == 1).count(),
                None => records,
            };
            if group_value_offset + batch_group_values > group_ids.len() {
                return Ok(None);
            }
            let batch_group_ids =
                &group_ids[group_value_offset..group_value_offset + batch_group_values];
            metrics.batches += 1;
            metrics.rows = metrics.rows.saturating_add(records);
            let consume_started = Instant::now();
            if consume(
                &predicate_values,
                &sum_values,
                batch_group_levels,
                batch_group_ids,
                &dictionary,
            )?
            .is_none()
            {
                return Ok(None);
            }
            metrics.add_consume_nanos(elapsed_nanos(consume_started));
            row_offset += records;
            group_value_offset += batch_group_values;
        }
        if row_offset != row_group.metadata().num_rows() as usize
            || group_value_offset != group_ids.len()
        {
            return Ok(None);
        }
    }
    Ok(Some(metrics))
}

fn scan_parquet_i32_dictionary_id_columns_reader<R, F>(
    reader: SerializedFileReader<R>,
    batch_size: usize,
    row_groups: &[usize],
    columns: [&str; 2],
    mut consume: F,
) -> Result<Option<DirectColumnScanMetrics>>
where
    R: ChunkReader + 'static,
    F: FnMut(&[i32], Option<&[i16]>, &[i32], &[Bytes]) -> Result<Option<()>>,
{
    let Some(column_indices) = parquet_column_indices_by_name(&reader, &columns) else {
        return Ok(None);
    };
    let [numeric_column, dictionary_column] =
        <[usize; 2]>::try_from(column_indices).map_err(|_| {
            DodamError::UnsupportedSql("direct parquet column index shape mismatch".to_string())
        })?;
    let schema = reader.metadata().file_metadata().schema_descr();
    let numeric_required = schema.column(numeric_column).max_def_level() == 0;
    let mut metrics = DirectColumnScanMetrics {
        row_groups: row_groups.len(),
        ..DirectColumnScanMetrics::default()
    };
    for &row_group_index in row_groups {
        let row_group = reader.get_row_group(row_group_index)?;
        let read_started = Instant::now();
        let Some((dictionary_def_levels, dictionary_ids, dictionary)) =
            read_byte_array_dictionary_ids_for_row_group(&*row_group, dictionary_column)?
        else {
            return Ok(None);
        };
        metrics.add_read_nanos(elapsed_nanos(read_started));

        let mut numeric_reader = match row_group.get_column_reader(numeric_column)? {
            ColumnReader::Int32ColumnReader(reader) => reader,
            _ => return Ok(None),
        };
        let mut numeric_values = Vec::<i32>::with_capacity(batch_size);
        let mut numeric_def_levels = Vec::<i16>::with_capacity(batch_size);
        let mut row_offset = 0usize;
        let mut dictionary_value_offset = 0usize;
        loop {
            numeric_values.clear();
            numeric_def_levels.clear();
            let read_started = Instant::now();
            let (records, numeric_value_count, _) = numeric_reader.read_records(
                batch_size,
                (!numeric_required).then_some(&mut numeric_def_levels),
                None,
                &mut numeric_values,
            )?;
            metrics.add_read_nanos(elapsed_nanos(read_started));
            if records == 0 {
                break;
            }
            if numeric_value_count != records
                || (!numeric_required && !direct_all_present(false, &numeric_def_levels))
                || row_offset + records > row_group.metadata().num_rows() as usize
            {
                return Ok(None);
            }
            let batch_dictionary_levels = if dictionary_def_levels.is_empty() {
                None
            } else {
                Some(&dictionary_def_levels[row_offset..row_offset + records])
            };
            let batch_dictionary_values = match batch_dictionary_levels {
                Some(levels) => levels.iter().filter(|level| **level == 1).count(),
                None => records,
            };
            if dictionary_value_offset + batch_dictionary_values > dictionary_ids.len() {
                return Ok(None);
            }
            let batch_dictionary_ids = &dictionary_ids
                [dictionary_value_offset..dictionary_value_offset + batch_dictionary_values];
            metrics.batches += 1;
            metrics.rows = metrics.rows.saturating_add(records);
            let consume_started = Instant::now();
            if consume(
                &numeric_values,
                batch_dictionary_levels,
                batch_dictionary_ids,
                &dictionary,
            )?
            .is_none()
            {
                return Ok(None);
            }
            metrics.add_consume_nanos(elapsed_nanos(consume_started));
            row_offset += records;
            dictionary_value_offset += batch_dictionary_values;
        }
        if row_offset != row_group.metadata().num_rows() as usize
            || dictionary_value_offset != dictionary_ids.len()
        {
            return Ok(None);
        }
    }
    Ok(Some(metrics))
}

fn read_byte_array_dictionary_ids_for_row_group(
    row_group: &dyn parquet::file::reader::RowGroupReader,
    column: usize,
) -> Result<Option<(Vec<i16>, Vec<i32>, Vec<Bytes>)>> {
    let column_desc = row_group.metadata().schema_descr().column(column);
    if column_desc.physical_type() != ParquetPhysicalType::BYTE_ARRAY
        || column_desc.max_rep_level() != 0
    {
        return Ok(None);
    }
    if column_desc.max_def_level() > 1 {
        return Ok(None);
    }
    let mut page_reader = row_group.get_column_page_reader(column)?;
    let mut dictionary: Option<Vec<Bytes>> = None;
    let mut def_levels = Vec::<i16>::new();
    let mut ids = Vec::<i32>::new();
    while let Some(page) = page_reader.get_next_page()? {
        match page {
            Page::DictionaryPage {
                buf,
                num_values,
                encoding,
                ..
            } => {
                if encoding != Encoding::PLAIN || dictionary.is_some() {
                    return Ok(None);
                }
                dictionary = Some(decode_plain_byte_array_dictionary(
                    buf,
                    num_values as usize,
                )?);
            }
            Page::DataPage {
                buf,
                num_values,
                encoding,
                def_level_encoding,
                ..
            } => {
                if !matches!(
                    encoding,
                    Encoding::RLE_DICTIONARY | Encoding::PLAIN_DICTIONARY
                ) {
                    return Ok(None);
                }
                let Some(dictionary) = dictionary.as_ref() else {
                    return Ok(None);
                };
                let mut offset = 0usize;
                let values = if column_desc.max_def_level() > 0 {
                    let (bytes_read, level_data) = parse_v1_rle_level_data(buf.slice(offset..))?;
                    offset += bytes_read;
                    if def_level_encoding != Encoding::RLE {
                        return Ok(None);
                    }
                    let start = def_levels.len();
                    decode_rle_i16_values(
                        level_data,
                        num_required_bits_i16(column_desc.max_def_level()),
                        num_values as usize,
                        &mut def_levels,
                    )?;
                    def_levels[start..]
                        .iter()
                        .filter(|level| **level == column_desc.max_def_level())
                        .count()
                } else {
                    num_values as usize
                };
                decode_dictionary_indices(buf.slice(offset..), values, dictionary.len(), &mut ids)?;
            }
            Page::DataPageV2 {
                buf,
                num_values,
                encoding,
                num_nulls,
                def_levels_byte_len,
                rep_levels_byte_len,
                ..
            } => {
                if !matches!(
                    encoding,
                    Encoding::RLE_DICTIONARY | Encoding::PLAIN_DICTIONARY
                ) {
                    return Ok(None);
                }
                let Some(dictionary) = dictionary.as_ref() else {
                    return Ok(None);
                };
                let values = if column_desc.max_def_level() > 0 {
                    let def_start = rep_levels_byte_len as usize;
                    let def_end = def_start + def_levels_byte_len as usize;
                    if def_end > buf.len() {
                        return Ok(None);
                    }
                    let start = def_levels.len();
                    decode_rle_i16_values(
                        buf.slice(def_start..def_end),
                        num_required_bits_i16(column_desc.max_def_level()),
                        num_values as usize,
                        &mut def_levels,
                    )?;
                    let values = def_levels[start..]
                        .iter()
                        .filter(|level| **level == column_desc.max_def_level())
                        .count();
                    if values != (num_values - num_nulls) as usize {
                        return Ok(None);
                    }
                    values
                } else {
                    num_values as usize
                };
                let value_start = (rep_levels_byte_len + def_levels_byte_len) as usize;
                if value_start > buf.len() {
                    return Ok(None);
                }
                decode_dictionary_indices(
                    buf.slice(value_start..),
                    values,
                    dictionary.len(),
                    &mut ids,
                )?;
            }
        }
    }
    let Some(dictionary) = dictionary else {
        return Ok(None);
    };
    if !def_levels.is_empty() && def_levels.len() != row_group.metadata().num_rows() as usize {
        return Ok(None);
    }
    Ok(Some((def_levels, ids, dictionary)))
}

fn read_i32_dictionary_ids_for_row_group(
    row_group: &dyn parquet::file::reader::RowGroupReader,
    column: usize,
) -> Result<Option<(Vec<i32>, Vec<i32>)>> {
    let column_desc = row_group.metadata().schema_descr().column(column);
    if column_desc.physical_type() != ParquetPhysicalType::INT32
        || column_desc.max_rep_level() != 0
        || column_desc.max_def_level() > 1
    {
        return Ok(None);
    }
    let row_count = usize::try_from(row_group.metadata().num_rows())
        .map_err(|_| DodamError::UnsupportedSql("row group row count out of range".to_string()))?;
    let mut page_reader = row_group.get_column_page_reader(column)?;
    let mut dictionary: Option<Vec<i32>> = None;
    let mut ids = Vec::<i32>::with_capacity(row_count);
    let mut rows = 0usize;
    while let Some(page) = page_reader.get_next_page()? {
        match page {
            Page::DictionaryPage {
                buf,
                num_values,
                encoding,
                ..
            } => {
                if encoding != Encoding::PLAIN || dictionary.is_some() {
                    return Ok(None);
                }
                let mut values = Vec::with_capacity(num_values as usize);
                decode_plain_i32_values(buf, num_values as usize, &mut values)?;
                if values.len() != num_values as usize {
                    return Ok(None);
                }
                dictionary = Some(values);
            }
            Page::DataPage {
                buf,
                num_values,
                encoding,
                def_level_encoding,
                ..
            } => {
                if !matches!(
                    encoding,
                    Encoding::RLE_DICTIONARY | Encoding::PLAIN_DICTIONARY
                ) {
                    return Ok(None);
                }
                let Some(dictionary) = dictionary.as_ref() else {
                    return Ok(None);
                };
                let mut offset = 0usize;
                let values = if column_desc.max_def_level() > 0 {
                    if def_level_encoding != Encoding::RLE {
                        return Ok(None);
                    }
                    let (bytes_read, level_data) = parse_v1_rle_level_data(buf.slice(offset..))?;
                    offset += bytes_read;
                    let mut def_levels = Vec::with_capacity(num_values as usize);
                    decode_rle_i16_values(
                        level_data,
                        num_required_bits_i16(column_desc.max_def_level()),
                        num_values as usize,
                        &mut def_levels,
                    )?;
                    if def_levels.len() != num_values as usize
                        || def_levels
                            .iter()
                            .any(|level| *level != column_desc.max_def_level())
                    {
                        return Ok(None);
                    }
                    num_values as usize
                } else {
                    num_values as usize
                };
                decode_dictionary_indices(buf.slice(offset..), values, dictionary.len(), &mut ids)?;
                rows = rows.saturating_add(values);
            }
            Page::DataPageV2 {
                buf,
                num_values,
                encoding,
                rep_levels_byte_len,
                def_levels_byte_len,
                ..
            } => {
                if !matches!(
                    encoding,
                    Encoding::RLE_DICTIONARY | Encoding::PLAIN_DICTIONARY
                ) {
                    return Ok(None);
                }
                let Some(dictionary) = dictionary.as_ref() else {
                    return Ok(None);
                };
                if column_desc.max_def_level() > 0 {
                    let def_start = rep_levels_byte_len as usize;
                    let def_end = def_start + def_levels_byte_len as usize;
                    if def_end > buf.len()
                        || !plain_page_all_present(
                            buf.slice(def_start..def_end),
                            column_desc.max_def_level(),
                            num_values as usize,
                        )?
                    {
                        return Ok(None);
                    }
                }
                let value_start = (rep_levels_byte_len + def_levels_byte_len) as usize;
                if value_start > buf.len() {
                    return Ok(None);
                }
                decode_dictionary_indices(
                    buf.slice(value_start..),
                    num_values as usize,
                    dictionary.len(),
                    &mut ids,
                )?;
                rows = rows.saturating_add(num_values as usize);
            }
        }
    }
    let Some(dictionary) = dictionary else {
        return Ok(None);
    };
    if rows != row_count || ids.len() != row_count {
        return Ok(None);
    }
    Ok(Some((ids, dictionary)))
}

fn build_i32x3_dictionary_selected_runs_for_row_group_pagewise<P>(
    row_group: &dyn parquet::file::reader::RowGroupReader,
    columns: [usize; 3],
    row_count: usize,
    predicate: &P,
    runs: &mut Vec<(usize, usize)>,
    builder: &mut SelectionRunsBuilder,
) -> Result<Option<()>>
where
    P: Fn(i32, i32, i32) -> bool,
{
    let schema = row_group.metadata().schema_descr();
    for column in columns {
        let column_desc = schema.column(column);
        if column_desc.physical_type() != ParquetPhysicalType::INT32
            || column_desc.max_rep_level() != 0
            || column_desc.max_def_level() > 1
        {
            return Ok(None);
        }
    }
    let mut first = I32DictionaryPageCursor::new(
        row_group.get_column_page_reader(columns[0])?,
        schema.column(columns[0]).max_def_level(),
    );
    let mut second = I32DictionaryPageCursor::new(
        row_group.get_column_page_reader(columns[1])?,
        schema.column(columns[1]).max_def_level(),
    );
    let mut third = I32DictionaryPageCursor::new(
        row_group.get_column_page_reader(columns[2])?,
        schema.column(columns[2]).max_def_level(),
    );
    let mut row_offset = 0usize;
    let mut run_start = None::<usize>;
    let mut run_len = 0usize;
    while row_offset < row_count {
        if !first.ensure_page()? || !second.ensure_page()? || !third.ensure_page()? {
            return Ok(None);
        }
        let records = (row_count - row_offset)
            .min(first.available_rows())
            .min(second.available_rows())
            .min(third.available_rows());
        if records == 0 {
            return Ok(None);
        }
        let first_ids = &first.ids[first.offset..first.offset + records];
        let second_ids = &second.ids[second.offset..second.offset + records];
        let third_ids = &third.ids[third.offset..third.offset + records];
        let Some(first_dictionary) = first.dictionary.as_deref() else {
            return Ok(None);
        };
        let Some(second_dictionary) = second.dictionary.as_deref() else {
            return Ok(None);
        };
        let Some(third_dictionary) = third.dictionary.as_deref() else {
            return Ok(None);
        };
        for (local, ((&first_id, &second_id), &third_id)) in first_ids
            .iter()
            .zip(second_ids.iter())
            .zip(third_ids.iter())
            .enumerate()
        {
            let first_idx = first_id as usize;
            let second_idx = second_id as usize;
            let third_idx = third_id as usize;
            // Dictionary ids are validated by decode_dictionary_indices before this hot loop.
            let selected = unsafe {
                predicate(
                    *first_dictionary.get_unchecked(first_idx),
                    *second_dictionary.get_unchecked(second_idx),
                    *third_dictionary.get_unchecked(third_idx),
                )
            };
            if selected {
                if run_start.is_none() {
                    run_start = Some(row_offset + local);
                    run_len = 1;
                } else {
                    run_len += 1;
                }
            } else if let Some(start) = run_start.take() {
                builder.push_disjoint_run(runs, start, run_len);
                run_len = 0;
            }
        }
        first.advance(records);
        second.advance(records);
        third.advance(records);
        row_offset += records;
    }
    if let Some(start) = run_start {
        builder.push_disjoint_run(runs, start, run_len);
    }
    Ok(Some(()))
}

struct I32DictionaryPageCursor {
    page_reader: Box<dyn PageReader>,
    max_def_level: i16,
    dictionary: Option<Vec<i32>>,
    ids: Vec<i32>,
    offset: usize,
}

impl I32DictionaryPageCursor {
    fn new(page_reader: Box<dyn PageReader>, max_def_level: i16) -> Self {
        Self {
            page_reader,
            max_def_level,
            dictionary: None,
            ids: Vec::new(),
            offset: 0,
        }
    }

    fn ensure_page(&mut self) -> Result<bool> {
        if self.available_rows() > 0 {
            return Ok(true);
        }
        self.load_next_data_page()
    }

    fn available_rows(&self) -> usize {
        self.ids.len().saturating_sub(self.offset)
    }

    fn advance(&mut self, records: usize) {
        self.offset += records;
        if self.offset >= self.ids.len() {
            self.ids.clear();
            self.offset = 0;
        }
    }

    fn load_next_data_page(&mut self) -> Result<bool> {
        while let Some(page) = self.page_reader.get_next_page()? {
            match page {
                Page::DictionaryPage {
                    buf,
                    num_values,
                    encoding,
                    ..
                } => {
                    if encoding != Encoding::PLAIN || self.dictionary.is_some() {
                        return Ok(false);
                    }
                    let mut values = Vec::with_capacity(num_values as usize);
                    decode_plain_i32_values(buf, num_values as usize, &mut values)?;
                    if values.len() != num_values as usize {
                        return Ok(false);
                    }
                    self.dictionary = Some(values);
                }
                Page::DataPage {
                    buf,
                    num_values,
                    encoding,
                    def_level_encoding,
                    ..
                } => {
                    if !matches!(
                        encoding,
                        Encoding::RLE_DICTIONARY | Encoding::PLAIN_DICTIONARY
                    ) {
                        return Ok(false);
                    }
                    let Some(dictionary) = self.dictionary.as_ref() else {
                        return Ok(false);
                    };
                    let mut offset = 0usize;
                    let values = if self.max_def_level > 0 {
                        if def_level_encoding != Encoding::RLE {
                            return Ok(false);
                        }
                        let (bytes_read, level_data) =
                            parse_v1_rle_level_data(buf.slice(offset..))?;
                        offset += bytes_read;
                        if !plain_page_all_present(
                            level_data,
                            self.max_def_level,
                            num_values as usize,
                        )? {
                            return Ok(false);
                        }
                        num_values as usize
                    } else {
                        num_values as usize
                    };
                    self.ids.clear();
                    decode_dictionary_indices(
                        buf.slice(offset..),
                        values,
                        dictionary.len(),
                        &mut self.ids,
                    )?;
                    if self.ids.len() != values {
                        return Ok(false);
                    }
                    self.offset = 0;
                    return Ok(true);
                }
                Page::DataPageV2 {
                    buf,
                    num_values,
                    encoding,
                    num_nulls,
                    rep_levels_byte_len,
                    def_levels_byte_len,
                    ..
                } => {
                    if !matches!(
                        encoding,
                        Encoding::RLE_DICTIONARY | Encoding::PLAIN_DICTIONARY
                    ) {
                        return Ok(false);
                    }
                    let Some(dictionary) = self.dictionary.as_ref() else {
                        return Ok(false);
                    };
                    if self.max_def_level > 0 && num_nulls > 0 {
                        let def_start = rep_levels_byte_len as usize;
                        let def_end = def_start + def_levels_byte_len as usize;
                        if def_end > buf.len()
                            || !plain_page_all_present(
                                buf.slice(def_start..def_end),
                                self.max_def_level,
                                num_values as usize,
                            )?
                        {
                            return Ok(false);
                        }
                    }
                    let value_start = (rep_levels_byte_len + def_levels_byte_len) as usize;
                    if value_start > buf.len() {
                        return Ok(false);
                    }
                    self.ids.clear();
                    decode_dictionary_indices(
                        buf.slice(value_start..),
                        num_values as usize,
                        dictionary.len(),
                        &mut self.ids,
                    )?;
                    if self.ids.len() != num_values as usize {
                        return Ok(false);
                    }
                    self.offset = 0;
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }
}

fn read_i64_dictionary_ids_for_row_group(
    row_group: &dyn parquet::file::reader::RowGroupReader,
    column: usize,
) -> Result<Option<(Vec<i32>, Vec<i64>)>> {
    let column_desc = row_group.metadata().schema_descr().column(column);
    if column_desc.physical_type() != ParquetPhysicalType::INT64
        || column_desc.max_rep_level() != 0
        || column_desc.max_def_level() > 1
    {
        return Ok(None);
    }
    let row_count = usize::try_from(row_group.metadata().num_rows())
        .map_err(|_| DodamError::UnsupportedSql("row group row count out of range".to_string()))?;
    let mut page_reader = row_group.get_column_page_reader(column)?;
    let mut dictionary: Option<Vec<i64>> = None;
    let mut ids = Vec::<i32>::with_capacity(row_count);
    let mut rows = 0usize;
    while let Some(page) = page_reader.get_next_page()? {
        match page {
            Page::DictionaryPage {
                buf,
                num_values,
                encoding,
                ..
            } => {
                if encoding != Encoding::PLAIN || dictionary.is_some() {
                    return Ok(None);
                }
                let mut values = Vec::with_capacity(num_values as usize);
                decode_plain_i64_values(buf, num_values as usize, &mut values)?;
                if values.len() != num_values as usize {
                    return Ok(None);
                }
                dictionary = Some(values);
            }
            Page::DataPage {
                buf,
                num_values,
                encoding,
                def_level_encoding,
                ..
            } => {
                if !matches!(
                    encoding,
                    Encoding::RLE_DICTIONARY | Encoding::PLAIN_DICTIONARY
                ) {
                    return Ok(None);
                }
                let Some(dictionary) = dictionary.as_ref() else {
                    return Ok(None);
                };
                let mut offset = 0usize;
                let values = if column_desc.max_def_level() > 0 {
                    if def_level_encoding != Encoding::RLE {
                        return Ok(None);
                    }
                    let (bytes_read, level_data) = parse_v1_rle_level_data(buf.slice(offset..))?;
                    offset += bytes_read;
                    let mut def_levels = Vec::with_capacity(num_values as usize);
                    decode_rle_i16_values(
                        level_data,
                        num_required_bits_i16(column_desc.max_def_level()),
                        num_values as usize,
                        &mut def_levels,
                    )?;
                    if def_levels.len() != num_values as usize
                        || def_levels
                            .iter()
                            .any(|level| *level != column_desc.max_def_level())
                    {
                        return Ok(None);
                    }
                    num_values as usize
                } else {
                    num_values as usize
                };
                decode_dictionary_indices(buf.slice(offset..), values, dictionary.len(), &mut ids)?;
                rows = rows.saturating_add(values);
            }
            Page::DataPageV2 {
                buf,
                num_values,
                encoding,
                rep_levels_byte_len,
                def_levels_byte_len,
                ..
            } => {
                if !matches!(
                    encoding,
                    Encoding::RLE_DICTIONARY | Encoding::PLAIN_DICTIONARY
                ) {
                    return Ok(None);
                }
                let Some(dictionary) = dictionary.as_ref() else {
                    return Ok(None);
                };
                if column_desc.max_def_level() > 0 {
                    let def_start = rep_levels_byte_len as usize;
                    let def_end = def_start + def_levels_byte_len as usize;
                    if def_end > buf.len()
                        || !plain_page_all_present(
                            buf.slice(def_start..def_end),
                            column_desc.max_def_level(),
                            num_values as usize,
                        )?
                    {
                        return Ok(None);
                    }
                }
                let value_start = (rep_levels_byte_len + def_levels_byte_len) as usize;
                if value_start > buf.len() {
                    return Ok(None);
                }
                decode_dictionary_indices(
                    buf.slice(value_start..),
                    num_values as usize,
                    dictionary.len(),
                    &mut ids,
                )?;
                rows = rows.saturating_add(num_values as usize);
            }
        }
    }
    let Some(dictionary) = dictionary else {
        return Ok(None);
    };
    if rows != row_count || ids.len() != row_count {
        return Ok(None);
    }
    Ok(Some((ids, dictionary)))
}

fn read_byte_array_dictionary_ids_selected_for_row_group(
    row_group: &dyn parquet::file::reader::RowGroupReader,
    column: usize,
    selected_runs: &[(usize, usize)],
    fallback: &[u8],
) -> Result<Option<(Vec<i32>, Vec<Bytes>)>> {
    let column_desc = row_group.metadata().schema_descr().column(column);
    if column_desc.physical_type() != ParquetPhysicalType::BYTE_ARRAY
        || column_desc.max_rep_level() != 0
        || column_desc.max_def_level() > 1
    {
        return Ok(None);
    }
    let mut page_reader = row_group.get_column_page_reader(column)?;
    let mut dictionary: Option<Vec<Bytes>> = None;
    let mut selected_ids = Vec::<i32>::new();
    let mut page_row_start = 0usize;
    let mut run_cursor = 0usize;
    while let Some(page) = page_reader.get_next_page()? {
        match page {
            Page::DictionaryPage {
                buf,
                num_values,
                encoding,
                ..
            } => {
                if encoding != Encoding::PLAIN || dictionary.is_some() {
                    return Ok(None);
                }
                dictionary = Some(decode_plain_byte_array_dictionary(
                    buf,
                    num_values as usize,
                )?);
            }
            Page::DataPage {
                buf,
                num_values,
                encoding,
                def_level_encoding,
                ..
            } => {
                if !matches!(
                    encoding,
                    Encoding::RLE_DICTIONARY | Encoding::PLAIN_DICTIONARY
                ) {
                    return Ok(None);
                }
                let Some(dictionary) = dictionary.as_ref() else {
                    return Ok(None);
                };
                let page_rows = num_values as usize;
                let page_row_end = page_row_start + page_rows;
                advance_run_cursor(selected_runs, &mut run_cursor, page_row_start);
                if !runs_overlap_from(selected_runs, run_cursor, page_row_start, page_row_end) {
                    page_row_start = page_row_end;
                    continue;
                }
                let mut offset = 0usize;
                let mut page_def_levels = Vec::<i16>::new();
                let values = if column_desc.max_def_level() > 0 {
                    let (bytes_read, level_data) = parse_v1_rle_level_data(buf.slice(offset..))?;
                    offset += bytes_read;
                    if def_level_encoding != Encoding::RLE {
                        return Ok(None);
                    }
                    decode_rle_i16_values(
                        level_data,
                        num_required_bits_i16(column_desc.max_def_level()),
                        page_rows,
                        &mut page_def_levels,
                    )?;
                    page_def_levels
                        .iter()
                        .filter(|level| **level == column_desc.max_def_level())
                        .count()
                } else {
                    page_rows
                };
                if page_def_levels.is_empty() {
                    decode_dictionary_indices_selected_ranges(
                        buf.slice(offset..),
                        values,
                        dictionary.len(),
                        &selected_runs[run_cursor..],
                        page_row_start,
                        page_rows,
                        &mut selected_ids,
                    )?;
                } else {
                    decode_dictionary_indices_selected_nullable_ranges(
                        buf.slice(offset..),
                        values,
                        dictionary.len(),
                        &selected_runs[run_cursor..],
                        page_row_start,
                        page_rows,
                        &page_def_levels,
                        column_desc.max_def_level(),
                        &mut selected_ids,
                    )?;
                }
                page_row_start = page_row_end;
            }
            Page::DataPageV2 {
                buf,
                num_values,
                encoding,
                num_nulls,
                def_levels_byte_len,
                rep_levels_byte_len,
                ..
            } => {
                if !matches!(
                    encoding,
                    Encoding::RLE_DICTIONARY | Encoding::PLAIN_DICTIONARY
                ) {
                    return Ok(None);
                }
                let Some(dictionary) = dictionary.as_ref() else {
                    return Ok(None);
                };
                let page_rows = num_values as usize;
                let page_row_end = page_row_start + page_rows;
                advance_run_cursor(selected_runs, &mut run_cursor, page_row_start);
                if !runs_overlap_from(selected_runs, run_cursor, page_row_start, page_row_end) {
                    page_row_start = page_row_end;
                    continue;
                }
                let mut page_def_levels = Vec::<i16>::new();
                let values = if column_desc.max_def_level() > 0 {
                    let def_start = rep_levels_byte_len as usize;
                    let def_end = def_start + def_levels_byte_len as usize;
                    if def_end > buf.len() {
                        return Ok(None);
                    }
                    decode_rle_i16_values(
                        buf.slice(def_start..def_end),
                        num_required_bits_i16(column_desc.max_def_level()),
                        page_rows,
                        &mut page_def_levels,
                    )?;
                    let values = page_def_levels
                        .iter()
                        .filter(|level| **level == column_desc.max_def_level())
                        .count();
                    if values != (num_values - num_nulls) as usize {
                        return Ok(None);
                    }
                    values
                } else {
                    page_rows
                };
                let value_start = (rep_levels_byte_len + def_levels_byte_len) as usize;
                if value_start > buf.len() {
                    return Ok(None);
                }
                if page_def_levels.is_empty() {
                    decode_dictionary_indices_selected_ranges(
                        buf.slice(value_start..),
                        values,
                        dictionary.len(),
                        &selected_runs[run_cursor..],
                        page_row_start,
                        page_rows,
                        &mut selected_ids,
                    )?;
                } else {
                    decode_dictionary_indices_selected_nullable_ranges(
                        buf.slice(value_start..),
                        values,
                        dictionary.len(),
                        &selected_runs[run_cursor..],
                        page_row_start,
                        page_rows,
                        &page_def_levels,
                        column_desc.max_def_level(),
                        &mut selected_ids,
                    )?;
                }
                page_row_start = page_row_end;
            }
        }
    }
    let Some(mut dictionary) = dictionary else {
        return Ok(None);
    };
    let fallback_id = dictionary_fallback_id(&mut dictionary, fallback)?;
    for id in &mut selected_ids {
        if *id < 0 {
            *id = fallback_id;
        }
    }
    Ok(Some((selected_ids, dictionary)))
}

fn read_i64_dictionary_values_selected_for_row_group(
    row_group: &dyn parquet::file::reader::RowGroupReader,
    column: usize,
    selected_runs: &[(usize, usize)],
) -> Result<Option<Vec<i64>>> {
    let column_desc = row_group.metadata().schema_descr().column(column);
    if column_desc.physical_type() != ParquetPhysicalType::INT64
        || column_desc.max_rep_level() != 0
        || column_desc.max_def_level() > 1
    {
        return Ok(None);
    }
    let mut page_reader = row_group.get_column_page_reader(column)?;
    let mut dictionary: Option<Vec<i64>> = None;
    let mut selected_ids = Vec::<i32>::new();
    let mut page_row_start = 0usize;
    let mut run_cursor = 0usize;
    while let Some(page) = page_reader.get_next_page()? {
        match page {
            Page::DictionaryPage {
                buf,
                num_values,
                encoding,
                ..
            } => {
                if encoding != Encoding::PLAIN || dictionary.is_some() {
                    return Ok(None);
                }
                let mut values = Vec::with_capacity(num_values as usize);
                decode_plain_i64_values(buf, num_values as usize, &mut values)?;
                if values.len() != num_values as usize {
                    return Ok(None);
                }
                dictionary = Some(values);
            }
            Page::DataPage {
                buf,
                num_values,
                encoding,
                def_level_encoding,
                ..
            } => {
                if !matches!(
                    encoding,
                    Encoding::RLE_DICTIONARY | Encoding::PLAIN_DICTIONARY
                ) {
                    return Ok(None);
                }
                let Some(dictionary) = dictionary.as_ref() else {
                    return Ok(None);
                };
                let page_rows = num_values as usize;
                let page_row_end = page_row_start + page_rows;
                advance_run_cursor(selected_runs, &mut run_cursor, page_row_start);
                if !runs_overlap_from(selected_runs, run_cursor, page_row_start, page_row_end) {
                    page_row_start = page_row_end;
                    continue;
                }
                let mut offset = 0usize;
                let mut page_def_levels = Vec::<i16>::new();
                let values = if column_desc.max_def_level() > 0 {
                    let (bytes_read, level_data) = parse_v1_rle_level_data(buf.slice(offset..))?;
                    offset += bytes_read;
                    if def_level_encoding != Encoding::RLE {
                        return Ok(None);
                    }
                    decode_rle_i16_values(
                        level_data,
                        num_required_bits_i16(column_desc.max_def_level()),
                        page_rows,
                        &mut page_def_levels,
                    )?;
                    page_def_levels
                        .iter()
                        .filter(|level| **level == column_desc.max_def_level())
                        .count()
                } else {
                    page_rows
                };
                if page_def_levels.is_empty() {
                    decode_dictionary_indices_selected_ranges(
                        buf.slice(offset..),
                        values,
                        dictionary.len(),
                        &selected_runs[run_cursor..],
                        page_row_start,
                        page_rows,
                        &mut selected_ids,
                    )?;
                } else {
                    decode_dictionary_indices_selected_nullable_ranges(
                        buf.slice(offset..),
                        values,
                        dictionary.len(),
                        &selected_runs[run_cursor..],
                        page_row_start,
                        page_rows,
                        &page_def_levels,
                        column_desc.max_def_level(),
                        &mut selected_ids,
                    )?;
                }
                page_row_start = page_row_end;
            }
            Page::DataPageV2 {
                buf,
                num_values,
                encoding,
                num_nulls,
                def_levels_byte_len,
                rep_levels_byte_len,
                ..
            } => {
                if !matches!(
                    encoding,
                    Encoding::RLE_DICTIONARY | Encoding::PLAIN_DICTIONARY
                ) {
                    return Ok(None);
                }
                let Some(dictionary) = dictionary.as_ref() else {
                    return Ok(None);
                };
                let page_rows = num_values as usize;
                let page_row_end = page_row_start + page_rows;
                advance_run_cursor(selected_runs, &mut run_cursor, page_row_start);
                if !runs_overlap_from(selected_runs, run_cursor, page_row_start, page_row_end) {
                    page_row_start = page_row_end;
                    continue;
                }
                let mut page_def_levels = Vec::<i16>::new();
                let values = if column_desc.max_def_level() > 0 {
                    let def_start = rep_levels_byte_len as usize;
                    let def_end = def_start + def_levels_byte_len as usize;
                    if def_end > buf.len() {
                        return Ok(None);
                    }
                    decode_rle_i16_values(
                        buf.slice(def_start..def_end),
                        num_required_bits_i16(column_desc.max_def_level()),
                        page_rows,
                        &mut page_def_levels,
                    )?;
                    let values = page_def_levels
                        .iter()
                        .filter(|level| **level == column_desc.max_def_level())
                        .count();
                    if values != (num_values - num_nulls) as usize {
                        return Ok(None);
                    }
                    values
                } else {
                    page_rows
                };
                let value_start = (rep_levels_byte_len + def_levels_byte_len) as usize;
                if value_start > buf.len() {
                    return Ok(None);
                }
                if page_def_levels.is_empty() {
                    decode_dictionary_indices_selected_ranges(
                        buf.slice(value_start..),
                        values,
                        dictionary.len(),
                        &selected_runs[run_cursor..],
                        page_row_start,
                        page_rows,
                        &mut selected_ids,
                    )?;
                } else {
                    decode_dictionary_indices_selected_nullable_ranges(
                        buf.slice(value_start..),
                        values,
                        dictionary.len(),
                        &selected_runs[run_cursor..],
                        page_row_start,
                        page_rows,
                        &page_def_levels,
                        column_desc.max_def_level(),
                        &mut selected_ids,
                    )?;
                }
                page_row_start = page_row_end;
            }
        }
    }
    let Some(dictionary) = dictionary else {
        return Ok(None);
    };
    let mut output = Vec::with_capacity(selected_ids.len());
    for id in selected_ids {
        let id = usize::try_from(id).map_err(|_| {
            DodamError::UnsupportedSql("negative selected i64 dictionary id".to_string())
        })?;
        let Some(value) = dictionary.get(id) else {
            return Ok(None);
        };
        output.push(*value);
    }
    Ok(Some(output))
}

fn build_i64_dictionary_selected_runs_for_row_group(
    row_group: &dyn parquet::file::reader::RowGroupReader,
    column: usize,
    records: usize,
    decimal_min: Option<i64>,
    decimal_max: Option<i64>,
    runs: &mut Vec<(usize, usize)>,
    builder: &mut SelectionRunsBuilder,
) -> Result<Option<usize>> {
    let column_desc = row_group.metadata().schema_descr().column(column);
    if column_desc.physical_type() != ParquetPhysicalType::INT64
        || column_desc.max_rep_level() != 0
        || column_desc.max_def_level() > 1
    {
        return Ok(None);
    }
    let mut page_reader = row_group.get_column_page_reader(column)?;
    let mut dictionary: Option<Vec<i64>> = None;
    let mut selected_dictionary_ids = Vec::<bool>::new();
    let mut def_levels = Vec::<i16>::new();
    let mut page_row_start = 0usize;
    while let Some(page) = page_reader.get_next_page()? {
        match page {
            Page::DictionaryPage {
                buf,
                num_values,
                encoding,
                ..
            } => {
                if encoding != Encoding::PLAIN || dictionary.is_some() {
                    return Ok(None);
                }
                let mut values = Vec::<i64>::with_capacity(num_values as usize);
                decode_plain_i64_values(buf, num_values as usize, &mut values)?;
                if values.len() != num_values as usize {
                    return Ok(None);
                }
                selected_dictionary_ids = values
                    .iter()
                    .map(|value| {
                        decimal_min.is_none_or(|min| *value >= min)
                            && decimal_max.is_none_or(|max| *value <= max)
                    })
                    .collect();
                dictionary = Some(values);
            }
            Page::DataPage {
                buf,
                num_values,
                encoding,
                def_level_encoding,
                ..
            } => {
                if !matches!(
                    encoding,
                    Encoding::RLE_DICTIONARY | Encoding::PLAIN_DICTIONARY
                ) {
                    return Ok(None);
                }
                let Some(dictionary) = dictionary.as_ref() else {
                    return Ok(None);
                };
                let page_rows = num_values as usize;
                let mut offset = 0usize;
                let present_values = if column_desc.max_def_level() > 0 {
                    let (bytes_read, level_data) = parse_v1_rle_level_data(buf.slice(offset..))?;
                    offset += bytes_read;
                    if def_level_encoding != Encoding::RLE {
                        return Ok(None);
                    }
                    def_levels.clear();
                    decode_rle_i16_values(
                        level_data,
                        num_required_bits_i16(column_desc.max_def_level()),
                        page_rows,
                        &mut def_levels,
                    )?;
                    def_levels
                        .iter()
                        .filter(|level| **level == column_desc.max_def_level())
                        .count()
                } else {
                    page_rows
                };
                append_dictionary_id_selected_runs_from_encoded(
                    buf.slice(offset..),
                    present_values,
                    dictionary.len(),
                    if def_levels.is_empty() {
                        None
                    } else {
                        Some((&def_levels, column_desc.max_def_level()))
                    },
                    &selected_dictionary_ids,
                    page_row_start,
                    page_rows,
                    runs,
                    builder,
                )?;
                page_row_start += page_rows;
            }
            Page::DataPageV2 {
                buf,
                num_values,
                encoding,
                num_nulls,
                def_levels_byte_len,
                rep_levels_byte_len,
                ..
            } => {
                if !matches!(
                    encoding,
                    Encoding::RLE_DICTIONARY | Encoding::PLAIN_DICTIONARY
                ) {
                    return Ok(None);
                }
                let Some(dictionary) = dictionary.as_ref() else {
                    return Ok(None);
                };
                let page_rows = num_values as usize;
                let present_values = if column_desc.max_def_level() > 0 && num_nulls != 0 {
                    let def_start = rep_levels_byte_len as usize;
                    let def_end = def_start + def_levels_byte_len as usize;
                    if def_end > buf.len() {
                        return Ok(None);
                    }
                    def_levels.clear();
                    decode_rle_i16_values(
                        buf.slice(def_start..def_end),
                        num_required_bits_i16(column_desc.max_def_level()),
                        page_rows,
                        &mut def_levels,
                    )?;
                    def_levels
                        .iter()
                        .filter(|level| **level == column_desc.max_def_level())
                        .count()
                } else {
                    def_levels.clear();
                    page_rows
                };
                let value_start = (rep_levels_byte_len + def_levels_byte_len) as usize;
                if value_start > buf.len() {
                    return Ok(None);
                }
                append_dictionary_id_selected_runs_from_encoded(
                    buf.slice(value_start..),
                    present_values,
                    dictionary.len(),
                    if def_levels.is_empty() {
                        None
                    } else {
                        Some((&def_levels, column_desc.max_def_level()))
                    },
                    &selected_dictionary_ids,
                    page_row_start,
                    page_rows,
                    runs,
                    builder,
                )?;
                page_row_start += page_rows;
            }
        }
    }
    if dictionary.is_none() || page_row_start != records {
        return Ok(None);
    }
    Ok(Some(builder.selected_rows()))
}

#[allow(clippy::too_many_arguments)]
fn append_dictionary_id_selected_runs_from_encoded(
    data: Bytes,
    values: usize,
    dictionary_len: usize,
    def_levels: Option<(&[i16], i16)>,
    selected_dictionary_ids: &[bool],
    row_offset: usize,
    rows: usize,
    runs: &mut Vec<(usize, usize)>,
    builder: &mut SelectionRunsBuilder,
) -> Result<()> {
    if data.is_empty() {
        return Ok(());
    }
    let bit_width = data[0];
    if bit_width > 32 {
        return Ok(());
    }
    if def_levels.is_none() {
        return append_dictionary_id_selected_runs_non_null_from_encoded_fast(
            data.slice(1..),
            bit_width,
            values,
            dictionary_len,
            selected_dictionary_ids,
            row_offset,
            rows,
            runs,
            builder,
        );
    }
    let mut decoder = SimpleRleBitpackedDecoder::new(data.slice(1..), bit_width);
    let mut decoded_block = DictionaryIdBlock::new();
    let mut decoded_block_pos = 0usize;
    let mut run_start = None;
    let mut run_len = 0usize;
    let mut value_index = 0usize;
    for row in 0..rows {
        let present = def_levels
            .map(|(levels, max_level)| levels[row] == max_level)
            .unwrap_or(true);
        let selected = if present {
            if decoded_block_pos >= decoded_block.len {
                if decoder.decode_next_block(&mut decoded_block)? == 0 {
                    return Ok(());
                }
                decoded_block_pos = 0;
            }
            let id = decoded_block.ids[decoded_block_pos];
            decoded_block_pos += 1;
            value_index += 1;
            if id as usize >= dictionary_len {
                return Ok(());
            }
            selected_dictionary_ids
                .get(id as usize)
                .copied()
                .unwrap_or(false)
        } else {
            false
        };
        if selected {
            if run_start.is_none() {
                run_start = Some(row_offset + row);
                run_len = 1;
            } else {
                run_len += 1;
            }
        } else if let Some(start) = run_start.take() {
            builder.push_run(runs, start, run_len);
            run_len = 0;
        }
    }
    if value_index != values {
        return Ok(());
    }
    if let Some(start) = run_start {
        builder.push_run(runs, start, run_len);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn append_dictionary_id_selected_runs_non_null_from_encoded_fast(
    data: Bytes,
    bit_width: u8,
    values: usize,
    dictionary_len: usize,
    selected_dictionary_ids: &[bool],
    row_offset: usize,
    rows: usize,
    runs: &mut Vec<(usize, usize)>,
    builder: &mut SelectionRunsBuilder,
) -> Result<()> {
    let mut pos = 0usize;
    let mut row = 0usize;
    let mut decoded = 0usize;
    let mut run_start = None;
    let mut run_len = 0usize;
    let mut block = DictionaryIdBlock::new();
    let selected_matcher = DictionaryIdMatcher::from_selected(selected_dictionary_ids);
    while decoded < values && row < rows {
        let Some(header) = read_hybrid_varint(&data, &mut pos)? else {
            return Ok(());
        };
        if header & 1 == 0 {
            let len = ((header >> 1) as usize)
                .min(values - decoded)
                .min(rows - row);
            let bytes = usize::from(bit_width).div_ceil(8);
            if pos + bytes > data.len() {
                return Ok(());
            }
            let mut id = 0u32;
            for index in 0..bytes {
                id |= u32::from(data[pos + index]) << (index * 8);
            }
            pos += bytes;
            if id as usize >= dictionary_len {
                return Ok(());
            }
            let selected = selected_matcher.contains(id);
            if selected {
                if run_start.is_none() {
                    run_start = Some(row_offset + row);
                    run_len = len;
                } else {
                    run_len += len;
                }
            } else if let Some(start) = run_start.take() {
                builder.push_run(runs, start, run_len);
                run_len = 0;
            }
            row += len;
            decoded += len;
        } else {
            let groups = (header >> 1) as usize;
            let count = groups
                .saturating_mul(8)
                .min(values - decoded)
                .min(rows - row);
            let bytes = groups
                .saturating_mul(8)
                .saturating_mul(usize::from(bit_width))
                .div_ceil(8);
            if pos + bytes > data.len() {
                return Ok(());
            }
            let mut bitpacked_reader =
                BitpackedDictionaryIdBlockReader::new(&data, pos, pos + bytes, bit_width);
            let mut local = 0usize;
            while local < count {
                let block_len = DICTIONARY_ID_BLOCK.min(count - local);
                let mask = if let Some(mask) = bitpacked_reader.decode_selected_mask_block(
                    block_len,
                    dictionary_len,
                    &selected_matcher,
                )? {
                    mask
                } else {
                    bitpacked_reader.decode_block(block_len, &mut block)?;
                    let Some(mask) = dictionary_id_block_selected_mask_bitset(
                        &block,
                        dictionary_len,
                        &selected_matcher,
                    ) else {
                        return Ok(());
                    };
                    mask
                };
                append_selected_mask_as_runs(
                    row_offset + row,
                    block_len,
                    mask,
                    &mut run_start,
                    &mut run_len,
                    runs,
                    builder,
                );
                row += block_len;
                decoded += block_len;
                local += block_len;
            }
            pos += bytes;
        }
    }
    if decoded != values || row != rows {
        return Ok(());
    }
    if let Some(start) = run_start {
        builder.push_run(runs, start, run_len);
    }
    Ok(())
}

const DICTIONARY_ID_BLOCK: usize = 64;

struct DictionaryIdBlock {
    ids: [u32; DICTIONARY_ID_BLOCK],
    len: usize,
}

impl DictionaryIdBlock {
    fn new() -> Self {
        Self {
            ids: [0; DICTIONARY_ID_BLOCK],
            len: 0,
        }
    }

    fn clear(&mut self) {
        self.len = 0;
    }

    fn push(&mut self, value: u32) {
        self.ids[self.len] = value;
        self.len += 1;
    }

    fn as_slice(&self) -> &[u32] {
        &self.ids[..self.len]
    }
}

enum DictionaryIdMatcher {
    None,
    All { len: usize },
    Ranges { len: usize, ranges: Vec<(u32, u32)> },
    BitSet { words: Vec<u64> },
}

impl DictionaryIdMatcher {
    fn from_selected(selected: &[bool]) -> Self {
        let mut selected_count = 0usize;
        let mut ranges = Vec::<(u32, u32)>::new();
        let mut index = 0usize;
        while index < selected.len() {
            if !selected[index] {
                index += 1;
                continue;
            }
            let start = index;
            while index < selected.len() && selected[index] {
                index += 1;
            }
            selected_count += index - start;
            ranges.push((start as u32, index as u32));
        }
        if selected_count == 0 {
            return Self::None;
        }
        if selected_count == selected.len() {
            return Self::All {
                len: selected.len(),
            };
        }
        if ranges.len() <= 8 {
            return Self::Ranges {
                len: selected.len(),
                ranges,
            };
        }
        let mut words = vec![0u64; selected.len().div_ceil(64)];
        for (index, is_selected) in selected.iter().copied().enumerate() {
            if is_selected {
                words[index / 64] |= 1u64 << (index % 64);
            }
        }
        Self::BitSet { words }
    }

    fn contains(&self, id: u32) -> bool {
        let index = id as usize;
        match self {
            Self::None => false,
            Self::All { len } => index < *len,
            Self::Ranges { len, ranges } => {
                index < *len && ranges.iter().any(|(start, end)| id >= *start && id < *end)
            }
            Self::BitSet { words } => words
                .get(index / 64)
                .map(|word| (word >> (index % 64)) & 1 != 0)
                .unwrap_or(false),
        }
    }

    fn mask_ids(&self, ids: &[u32], base_lane: usize, dictionary_len: usize) -> Result<u64> {
        match self {
            Self::None => {
                for id in ids {
                    if *id as usize >= dictionary_len {
                        return Err(invalid_dictionary_id_error());
                    }
                }
                Ok(0)
            }
            Self::All { len } => {
                let mut mask = 0u64;
                for (offset, id) in ids.iter().copied().enumerate() {
                    if id as usize >= dictionary_len {
                        return Err(invalid_dictionary_id_error());
                    }
                    if (id as usize) < *len {
                        mask |= 1u64 << (base_lane + offset);
                    }
                }
                Ok(mask)
            }
            Self::Ranges { len, ranges } => {
                let mut mask = 0u64;
                if ranges.len() == 1 {
                    let (start, end) = ranges[0];
                    for (offset, id) in ids.iter().copied().enumerate() {
                        if id as usize >= dictionary_len {
                            return Err(invalid_dictionary_id_error());
                        }
                        if (id as usize) < *len && id >= start && id < end {
                            mask |= 1u64 << (base_lane + offset);
                        }
                    }
                    return Ok(mask);
                }
                for (offset, id) in ids.iter().copied().enumerate() {
                    if id as usize >= dictionary_len {
                        return Err(invalid_dictionary_id_error());
                    }
                    if (id as usize) < *len
                        && ranges.iter().any(|(start, end)| id >= *start && id < *end)
                    {
                        mask |= 1u64 << (base_lane + offset);
                    }
                }
                Ok(mask)
            }
            Self::BitSet { words } => {
                let mut mask = 0u64;
                for (offset, id) in ids.iter().copied().enumerate() {
                    let index = id as usize;
                    if index >= dictionary_len {
                        return Err(invalid_dictionary_id_error());
                    }
                    if words
                        .get(index / 64)
                        .map(|word| (word >> (index % 64)) & 1 != 0)
                        .unwrap_or(false)
                    {
                        mask |= 1u64 << (base_lane + offset);
                    }
                }
                Ok(mask)
            }
        }
    }
}

fn dictionary_id_block_selected_mask_bitset(
    ids: &DictionaryIdBlock,
    dictionary_len: usize,
    selected_dictionary_ids: &DictionaryIdMatcher,
) -> Option<u64> {
    let mut mask = 0u64;
    for (index, id) in ids.as_slice().iter().copied().enumerate() {
        if id as usize >= dictionary_len {
            return None;
        }
        if selected_dictionary_ids.contains(id) {
            mask |= 1u64 << index;
        }
    }
    Some(mask)
}

fn invalid_dictionary_id_error() -> DodamError {
    DodamError::Parquet(ParquetError::General(
        "parquet dictionary id out of range".to_string(),
    ))
}

fn append_selected_mask_as_runs(
    row_base: usize,
    len: usize,
    mask: u64,
    pending_start: &mut Option<usize>,
    pending_len: &mut usize,
    runs: &mut Vec<(usize, usize)>,
    builder: &mut SelectionRunsBuilder,
) {
    if len == 0 {
        return;
    }
    let valid_mask = if len >= 64 {
        u64::MAX
    } else {
        (1u64 << len) - 1
    };
    let mut selected = mask & valid_mask;
    let mut cursor = 0usize;
    while selected != 0 {
        let gap = selected.trailing_zeros() as usize;
        if gap > 0 {
            if let Some(start) = pending_start.take() {
                builder.push_run(runs, start, *pending_len);
                *pending_len = 0;
            }
            selected >>= gap;
            cursor += gap;
        }
        let run_len = selected.trailing_ones() as usize;
        if run_len == 0 {
            break;
        }
        if pending_start.is_none() {
            *pending_start = Some(row_base + cursor);
            *pending_len = run_len;
        } else {
            *pending_len += run_len;
        }
        if run_len >= 64 {
            selected = 0;
        } else {
            selected >>= run_len;
        }
        cursor += run_len;
    }
    if cursor < len {
        if let Some(start) = pending_start.take() {
            builder.push_run(runs, start, *pending_len);
            *pending_len = 0;
        }
    }
}

struct BitpackedDictionaryIdBlockReader<'a> {
    data: &'a Bytes,
    pos: usize,
    end: usize,
    bit_width: u8,
    bit_buffer: u64,
    buffered_bits: u8,
}

impl<'a> BitpackedDictionaryIdBlockReader<'a> {
    fn new(data: &'a Bytes, pos: usize, end: usize, bit_width: u8) -> Self {
        Self {
            data,
            pos,
            end,
            bit_width,
            bit_buffer: 0,
            buffered_bits: 0,
        }
    }

    fn decode_block(&mut self, len: usize, output: &mut DictionaryIdBlock) -> Result<()> {
        output.clear();
        if self.bit_width == 0 {
            for _ in 0..len {
                output.push(0);
            }
            return Ok(());
        }
        if self.buffered_bits == 0 {
            self.decode_aligned_chunks(len, output);
            if output.len == len {
                return Ok(());
            }
        }
        let mask = if self.bit_width == 32 {
            u64::from(u32::MAX)
        } else {
            (1_u64 << self.bit_width) - 1
        };
        while output.len < len {
            while self.buffered_bits < self.bit_width {
                if self.pos >= self.end {
                    return Err(DodamError::Parquet(ParquetError::General(
                        "not enough data to decode parquet dictionary id block".to_string(),
                    )));
                }
                self.bit_buffer |= u64::from(self.data[self.pos]) << self.buffered_bits;
                self.buffered_bits += 8;
                self.pos += 1;
            }
            output.push((self.bit_buffer & mask) as u32);
            self.bit_buffer >>= self.bit_width;
            self.buffered_bits -= self.bit_width;
        }
        Ok(())
    }

    fn decode_selected_mask_block(
        &mut self,
        len: usize,
        dictionary_len: usize,
        matcher: &DictionaryIdMatcher,
    ) -> Result<Option<u64>> {
        if self.buffered_bits != 0 || len > DICTIONARY_ID_BLOCK {
            return Ok(None);
        }
        match self.bit_width {
            8 => {
                if self.pos + len > self.end {
                    return Err(DodamError::Parquet(ParquetError::General(
                        "not enough data to decode parquet dictionary id block".to_string(),
                    )));
                }
                let mut mask = 0u64;
                for lane in 0..len {
                    let id = u32::from(self.data[self.pos + lane]);
                    mask |= matcher.mask_ids(std::slice::from_ref(&id), lane, dictionary_len)?;
                }
                self.pos += len;
                Ok(Some(mask))
            }
            14 => {
                if len % 4 != 0 {
                    return Ok(None);
                }
                let required = len / 4 * 7;
                if self.pos + required > self.end {
                    return Err(DodamError::Parquet(ParquetError::General(
                        "not enough data to decode parquet dictionary id block".to_string(),
                    )));
                }
                let mut mask = 0u64;
                let mut lane = 0usize;
                let end = self.pos + required;
                while self.pos < end {
                    let word = u64::from(self.data[self.pos])
                        | (u64::from(self.data[self.pos + 1]) << 8)
                        | (u64::from(self.data[self.pos + 2]) << 16)
                        | (u64::from(self.data[self.pos + 3]) << 24)
                        | (u64::from(self.data[self.pos + 4]) << 32)
                        | (u64::from(self.data[self.pos + 5]) << 40)
                        | (u64::from(self.data[self.pos + 6]) << 48);
                    let ids = [
                        (word & 0x3fff) as u32,
                        ((word >> 14) & 0x3fff) as u32,
                        ((word >> 28) & 0x3fff) as u32,
                        ((word >> 42) & 0x3fff) as u32,
                    ];
                    mask |= matcher.mask_ids(&ids, lane, dictionary_len)?;
                    lane += ids.len();
                    self.pos += 7;
                }
                Ok(Some(mask))
            }
            16 => {
                let required = len.saturating_mul(2);
                if self.pos + required > self.end {
                    return Err(DodamError::Parquet(ParquetError::General(
                        "not enough data to decode parquet dictionary id block".to_string(),
                    )));
                }
                let mut mask = 0u64;
                for lane in 0..len {
                    let offset = self.pos + lane * 2;
                    let id = u32::from(u16::from_le_bytes([
                        self.data[offset],
                        self.data[offset + 1],
                    ]));
                    mask |= matcher.mask_ids(std::slice::from_ref(&id), lane, dictionary_len)?;
                }
                self.pos += required;
                Ok(Some(mask))
            }
            32 => {
                let required = len.saturating_mul(4);
                if self.pos + required > self.end {
                    return Err(DodamError::Parquet(ParquetError::General(
                        "not enough data to decode parquet dictionary id block".to_string(),
                    )));
                }
                let mut mask = 0u64;
                for lane in 0..len {
                    let offset = self.pos + lane * 4;
                    let id = u32::from_le_bytes([
                        self.data[offset],
                        self.data[offset + 1],
                        self.data[offset + 2],
                        self.data[offset + 3],
                    ]);
                    mask |= matcher.mask_ids(std::slice::from_ref(&id), lane, dictionary_len)?;
                }
                self.pos += required;
                Ok(Some(mask))
            }
            _ => Ok(None),
        }
    }

    fn decode_aligned_chunks(&mut self, len: usize, output: &mut DictionaryIdBlock) {
        match self.bit_width {
            8 => {
                while output.len < len && self.pos < self.end {
                    output.push(u32::from(self.data[self.pos]));
                    self.pos += 1;
                }
            }
            14 => {
                while output.len + 4 <= len && self.pos + 7 <= self.end {
                    let word = u64::from(self.data[self.pos])
                        | (u64::from(self.data[self.pos + 1]) << 8)
                        | (u64::from(self.data[self.pos + 2]) << 16)
                        | (u64::from(self.data[self.pos + 3]) << 24)
                        | (u64::from(self.data[self.pos + 4]) << 32)
                        | (u64::from(self.data[self.pos + 5]) << 40)
                        | (u64::from(self.data[self.pos + 6]) << 48);
                    output.push((word & 0x3fff) as u32);
                    output.push(((word >> 14) & 0x3fff) as u32);
                    output.push(((word >> 28) & 0x3fff) as u32);
                    output.push(((word >> 42) & 0x3fff) as u32);
                    self.pos += 7;
                }
            }
            16 => {
                while output.len < len && self.pos + 2 <= self.end {
                    output.push(u32::from(u16::from_le_bytes([
                        self.data[self.pos],
                        self.data[self.pos + 1],
                    ])));
                    self.pos += 2;
                }
            }
            32 => {
                while output.len < len && self.pos + 4 <= self.end {
                    output.push(u32::from_le_bytes([
                        self.data[self.pos],
                        self.data[self.pos + 1],
                        self.data[self.pos + 2],
                        self.data[self.pos + 3],
                    ]));
                    self.pos += 4;
                }
            }
            _ => {}
        }
    }
}

fn read_hybrid_varint(data: &Bytes, pos: &mut usize) -> Result<Option<u64>> {
    let mut shift = 0u32;
    let mut value = 0u64;
    while *pos < data.len() {
        let byte = data[*pos];
        *pos += 1;
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Ok(Some(value));
        }
        shift += 7;
        if shift >= 64 {
            return Err(DodamError::Parquet(ParquetError::General(
                "parquet rle varint overflow".to_string(),
            )));
        }
    }
    Ok(None)
}

fn parse_v1_rle_level_data(buf: Bytes) -> Result<(usize, Bytes)> {
    if buf.len() < 4 {
        return Err(DodamError::Parquet(ParquetError::General(
            "not enough data to read parquet v1 level length".to_string(),
        )));
    }
    let len = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
    if buf.len() < 4 + len {
        return Err(DodamError::Parquet(ParquetError::General(
            "not enough data to read parquet v1 level data".to_string(),
        )));
    }
    Ok((4 + len, buf.slice(4..4 + len)))
}

fn decode_plain_byte_array_dictionary(buf: Bytes, num_values: usize) -> Result<Vec<Bytes>> {
    let mut offset = 0usize;
    let mut values = Vec::with_capacity(num_values);
    for _ in 0..num_values {
        if offset + 4 > buf.len() {
            return Ok(Vec::new());
        }
        let len = u32::from_le_bytes([
            buf[offset],
            buf[offset + 1],
            buf[offset + 2],
            buf[offset + 3],
        ]) as usize;
        offset += 4;
        if offset + len > buf.len() {
            return Ok(Vec::new());
        }
        values.push(buf.slice(offset..offset + len));
        offset += len;
    }
    Ok(values)
}

fn decode_dictionary_indices(
    data: Bytes,
    values: usize,
    dictionary_len: usize,
    output: &mut Vec<i32>,
) -> Result<()> {
    if data.is_empty() {
        return Ok(());
    }
    let bit_width = data[0];
    let before = output.len();
    decode_rle_i32_values(data.slice(1..), bit_width, values, output)?;
    if output.len() != before + values
        || output[before..]
            .iter()
            .any(|id| *id < 0 || *id as usize >= dictionary_len)
    {
        return Err(DodamError::UnsupportedSql(
            "dictionary id is out of range".to_string(),
        ));
    }
    Ok(())
}

fn decode_dictionary_indices_selected_ranges(
    data: Bytes,
    values: usize,
    dictionary_len: usize,
    selected_runs: &[(usize, usize)],
    page_row_start: usize,
    page_rows: usize,
    output: &mut Vec<i32>,
) -> Result<()> {
    if data.is_empty() {
        return Ok(());
    }
    let bit_width = data[0];
    if bit_width > 32 {
        return Ok(());
    }
    let mut decoder = SimpleRleBitpackedDecoder::new(data.slice(1..), bit_width);
    let page_row_end = page_row_start + page_rows;
    let mut cursor = 0usize;
    for &(run_start, run_len) in selected_runs {
        if run_start >= page_row_end {
            break;
        }
        let run_end = run_start + run_len;
        let start = run_start.max(page_row_start);
        let end = run_end.min(page_row_end);
        if start >= end {
            continue;
        }
        let local_start = start - page_row_start;
        let local_end = end - page_row_start;
        if local_start > values || local_end > values || cursor > local_start {
            return Ok(());
        }
        decoder.skip_values(local_start - cursor)?;
        cursor = local_start;
        output.reserve(local_end - local_start);
        for _ in local_start..local_end {
            let Some(value) = decoder.next_value()? else {
                return Ok(());
            };
            if value as usize >= dictionary_len {
                return Ok(());
            }
            output.push(value as i32);
            cursor += 1;
        }
    }
    Ok(())
}

fn decode_dictionary_indices_selected_nullable_ranges(
    data: Bytes,
    values: usize,
    dictionary_len: usize,
    selected_runs: &[(usize, usize)],
    page_row_start: usize,
    page_rows: usize,
    def_levels: &[i16],
    max_def_level: i16,
    output: &mut Vec<i32>,
) -> Result<()> {
    if data.is_empty() || def_levels.len() != page_rows {
        return Ok(());
    }
    let bit_width = data[0];
    if bit_width > 32 {
        return Ok(());
    }
    let mut decoder = SimpleRleBitpackedDecoder::new(data.slice(1..), bit_width);
    let page_row_end = page_row_start + page_rows;
    let mut cursor_row = 0usize;
    let mut decoded_values = 0usize;
    for &(run_start, run_len) in selected_runs {
        if run_start >= page_row_end {
            break;
        }
        let run_end = run_start + run_len;
        let start = run_start.max(page_row_start);
        let end = run_end.min(page_row_end);
        if start >= end {
            continue;
        }
        let local_start = start - page_row_start;
        let local_end = end - page_row_start;
        if cursor_row > local_start || local_end > page_rows {
            return Ok(());
        }
        let skip_present =
            count_present_def_levels(&def_levels[cursor_row..local_start], max_def_level);
        if decoded_values + skip_present > values {
            return Ok(());
        }
        decoder.skip_values(skip_present)?;
        decoded_values += skip_present;
        output.reserve(local_end - local_start);
        for row in local_start..local_end {
            if def_levels[row] == max_def_level {
                let Some(value) = decoder.next_value()? else {
                    return Ok(());
                };
                if value as usize >= dictionary_len {
                    return Ok(());
                }
                output.push(value as i32);
                decoded_values += 1;
            } else {
                output.push(-1);
            }
        }
        cursor_row = local_end;
    }
    Ok(())
}

fn count_present_def_levels(def_levels: &[i16], max_def_level: i16) -> usize {
    def_levels
        .iter()
        .filter(|level| **level == max_def_level)
        .count()
}

fn num_required_bits_i16(value: i16) -> u8 {
    let mut value = value as u16;
    let mut bits = 0u8;
    while value > 0 {
        bits += 1;
        value >>= 1;
    }
    bits
}

fn decode_rle_i16_values(
    data: Bytes,
    bit_width: u8,
    values: usize,
    output: &mut Vec<i16>,
) -> Result<()> {
    if bit_width > 16 {
        return Ok(());
    }
    let before = output.len();
    let mut decoder = SimpleRleBitpackedDecoder::new(data, bit_width);
    output.reserve(values);
    for _ in 0..values {
        let Some(value) = decoder.next_value()? else {
            break;
        };
        output.push(value as i16);
    }
    if output.len() != before + values {
        return Ok(());
    }
    Ok(())
}

fn rle_i16_all_equal(data: Bytes, bit_width: u8, values: usize, expected: i16) -> Result<bool> {
    if bit_width > 16 {
        return Ok(false);
    }
    let mut decoder = SimpleRleBitpackedDecoder::new(data, bit_width);
    let expected = u32::try_from(expected).map_err(|_| {
        DodamError::Parquet(ParquetError::General(
            "negative parquet definition level".to_string(),
        ))
    })?;
    for _ in 0..values {
        let Some(value) = decoder.next_value()? else {
            return Ok(false);
        };
        if value != expected {
            return Ok(false);
        }
    }
    Ok(true)
}

fn decode_rle_i32_values(
    data: Bytes,
    bit_width: u8,
    values: usize,
    output: &mut Vec<i32>,
) -> Result<()> {
    if bit_width > 32 {
        return Ok(());
    }
    trace_dictionary_bit_width(bit_width, values);
    let mut decoder = SimpleRleBitpackedDecoder::new(data, bit_width);
    output.reserve(values);
    decoder.decode_i32_batch(values, output)?;
    Ok(())
}

fn trace_dictionary_bit_width(bit_width: u8, values: usize) {
    if !std::env::var("DODAM_TRACE_DICTIONARY_BIT_WIDTH")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
    {
        return;
    }
    eprintln!("[dodam:dictionary-bit-width] width={bit_width} values={values}");
}

struct SimpleRleBitpackedDecoder {
    data: Bytes,
    bit_width: u8,
    pos: usize,
    rle_remaining: usize,
    rle_value: u32,
    bitpack_remaining: usize,
    bitpack_bit_offset: usize,
    bitpack_end: usize,
}

impl SimpleRleBitpackedDecoder {
    fn new(data: Bytes, bit_width: u8) -> Self {
        Self {
            data,
            bit_width,
            pos: 0,
            rle_remaining: 0,
            rle_value: 0,
            bitpack_remaining: 0,
            bitpack_bit_offset: 0,
            bitpack_end: 0,
        }
    }

    fn next_value(&mut self) -> Result<Option<u32>> {
        loop {
            if self.rle_remaining > 0 {
                self.rle_remaining -= 1;
                return Ok(Some(self.rle_value));
            }
            if self.bitpack_remaining > 0 {
                let value = self.read_bitpacked_value()?;
                self.bitpack_remaining -= 1;
                if self.bitpack_remaining == 0 {
                    self.pos = self.bitpack_end;
                }
                return Ok(Some(value));
            }
            if !self.reload_run()? {
                return Ok(None);
            }
        }
    }

    fn skip_values(&mut self, mut values: usize) -> Result<()> {
        while values > 0 {
            if self.rle_remaining > 0 {
                let skipped = self.rle_remaining.min(values);
                self.rle_remaining -= skipped;
                values -= skipped;
                continue;
            }
            if self.bitpack_remaining > 0 {
                let skipped = self.bitpack_remaining.min(values);
                self.bitpack_bit_offset = self
                    .bitpack_bit_offset
                    .saturating_add(skipped.saturating_mul(usize::from(self.bit_width)));
                self.bitpack_remaining -= skipped;
                values -= skipped;
                if self.bitpack_remaining == 0 {
                    self.pos = self.bitpack_end;
                }
                continue;
            }
            if !self.reload_run()? {
                return Ok(());
            }
        }
        Ok(())
    }

    fn decode_i32_batch(&mut self, mut values: usize, output: &mut Vec<i32>) -> Result<usize> {
        let mut decoded = 0usize;
        while values > 0 {
            if self.rle_remaining > 0 {
                let len = self.rle_remaining.min(values);
                output.extend(std::iter::repeat_n(self.rle_value as i32, len));
                self.rle_remaining -= len;
                values -= len;
                decoded += len;
                continue;
            }
            if self.bitpack_remaining > 0 {
                let len = self.bitpack_remaining.min(values);
                if !self.read_bitpacked_i32_batch_fast(len, output)? {
                    output.reserve(len);
                    for _ in 0..len {
                        output.push(self.read_bitpacked_value()? as i32);
                    }
                }
                self.bitpack_remaining -= len;
                values -= len;
                decoded += len;
                if self.bitpack_remaining == 0 {
                    self.pos = self.bitpack_end;
                }
                continue;
            }
            if !self.reload_run()? {
                break;
            }
        }
        Ok(decoded)
    }

    fn read_bitpacked_i32_batch_fast(&mut self, len: usize, output: &mut Vec<i32>) -> Result<bool> {
        if self.bitpack_bit_offset % 8 != 0 || len == 0 {
            return Ok(false);
        }
        let byte_pos = self.bitpack_bit_offset / 8;
        match self.bit_width {
            0 => {
                output.extend(std::iter::repeat_n(0, len));
                return Ok(true);
            }
            8 => {
                if byte_pos + len > self.bitpack_end {
                    return Ok(false);
                }
                output.reserve(len);
                output.extend(
                    self.data[byte_pos..byte_pos + len]
                        .iter()
                        .map(|value| i32::from(*value)),
                );
                self.bitpack_bit_offset += len * 8;
                return Ok(true);
            }
            16 => {
                let byte_len = len.saturating_mul(2);
                if byte_pos + byte_len > self.bitpack_end {
                    return Ok(false);
                }
                output.reserve(len);
                for chunk in self.data[byte_pos..byte_pos + byte_len].chunks_exact(2) {
                    output.push(i32::from(u16::from_le_bytes([chunk[0], chunk[1]])));
                }
                self.bitpack_bit_offset += byte_len * 8;
                return Ok(true);
            }
            12 => {
                let byte_len = len.saturating_mul(12).div_ceil(8);
                if byte_pos + byte_len > self.bitpack_end {
                    return Ok(false);
                }
                output.reserve(len);
                let mut pos = byte_pos;
                let pairs = len / 2;
                for _ in 0..pairs {
                    let b0 = u32::from(self.data[pos]);
                    let b1 = u32::from(self.data[pos + 1]);
                    let b2 = u32::from(self.data[pos + 2]);
                    output.push(((b0 | ((b1 & 0x0f) << 8)) & 0x0fff) as i32);
                    output.push((((b1 >> 4) | (b2 << 4)) & 0x0fff) as i32);
                    pos += 3;
                }
                if len % 2 != 0 {
                    let b0 = u32::from(self.data[pos]);
                    let b1 = u32::from(self.data[pos + 1]);
                    output.push(((b0 | ((b1 & 0x0f) << 8)) & 0x0fff) as i32);
                }
                self.bitpack_bit_offset += len * 12;
                return Ok(true);
            }
            32 => {
                let byte_len = len.saturating_mul(4);
                if byte_pos + byte_len > self.bitpack_end {
                    return Ok(false);
                }
                output.reserve(len);
                for chunk in self.data[byte_pos..byte_pos + byte_len].chunks_exact(4) {
                    output.push(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                }
                self.bitpack_bit_offset += byte_len * 8;
                return Ok(true);
            }
            _ => {}
        }
        Ok(false)
    }

    fn decode_next_block(&mut self, output: &mut DictionaryIdBlock) -> Result<usize> {
        output.clear();
        while output.len < DICTIONARY_ID_BLOCK {
            let Some(value) = self.next_value()? else {
                break;
            };
            output.push(value);
        }
        Ok(output.len)
    }

    fn reload_run(&mut self) -> Result<bool> {
        let Some(header) = self.read_varint()? else {
            return Ok(false);
        };
        if header & 1 == 0 {
            self.rle_remaining = (header >> 1) as usize;
            let bytes = usize::from(self.bit_width).div_ceil(8);
            if self.pos + bytes > self.data.len() {
                return Err(DodamError::Parquet(ParquetError::General(
                    "not enough data to read parquet rle value".to_string(),
                )));
            }
            let mut value = 0u32;
            for index in 0..bytes {
                value |= (self.data[self.pos + index] as u32) << (index * 8);
            }
            self.pos += bytes;
            self.rle_value = value;
        } else {
            let groups = (header >> 1) as usize;
            self.bitpack_remaining = groups.saturating_mul(8);
            self.bitpack_bit_offset = self.pos.saturating_mul(8);
            let bytes = self
                .bitpack_remaining
                .saturating_mul(usize::from(self.bit_width))
                .div_ceil(8);
            self.bitpack_end = self.pos.saturating_add(bytes);
            if self.bitpack_end > self.data.len() {
                return Err(DodamError::Parquet(ParquetError::General(
                    "not enough data to read parquet bit-packed values".to_string(),
                )));
            }
        }
        Ok(true)
    }

    fn read_varint(&mut self) -> Result<Option<u64>> {
        let mut shift = 0u32;
        let mut value = 0u64;
        while self.pos < self.data.len() {
            let byte = self.data[self.pos];
            self.pos += 1;
            value |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Ok(Some(value));
            }
            shift += 7;
            if shift >= 64 {
                return Err(DodamError::Parquet(ParquetError::General(
                    "parquet rle varint overflow".to_string(),
                )));
            }
        }
        Ok(None)
    }

    fn read_bitpacked_value(&mut self) -> Result<u32> {
        if self.bit_width == 0 {
            return Ok(0);
        }
        let absolute_bit = self.bitpack_bit_offset;
        let byte_index = absolute_bit / 8;
        let bit_index = absolute_bit % 8;
        let needed_bits = bit_index + usize::from(self.bit_width);
        if needed_bits <= 64 && byte_index < self.bitpack_end {
            let available = self.bitpack_end - byte_index;
            let bytes = available.min(8);
            if bytes.saturating_mul(8) >= needed_bits {
                let mut word = 0u64;
                for index in 0..bytes {
                    word |= u64::from(self.data[byte_index + index]) << (index * 8);
                }
                let mask = if self.bit_width == 32 {
                    u64::from(u32::MAX)
                } else {
                    (1_u64 << self.bit_width) - 1
                };
                self.bitpack_bit_offset += usize::from(self.bit_width);
                return Ok(((word >> bit_index) & mask) as u32);
            }
        }
        let mut value = 0u32;
        for bit in 0..usize::from(self.bit_width) {
            let absolute_bit = self.bitpack_bit_offset + bit;
            let byte_index = absolute_bit / 8;
            if byte_index >= self.bitpack_end {
                return Err(DodamError::Parquet(ParquetError::General(
                    "not enough data to read parquet bit-packed value".to_string(),
                )));
            }
            let bit_index = absolute_bit % 8;
            let bit_value = (self.data[byte_index] >> bit_index) & 1;
            value |= u32::from(bit_value) << bit;
        }
        self.bitpack_bit_offset += usize::from(self.bit_width);
        Ok(value)
    }
}

fn scan_parquet_i32_i64_byte_array_columns_reader<R, F>(
    reader: SerializedFileReader<R>,
    batch_size: usize,
    row_groups: &[usize],
    columns: [&str; 3],
    mut consume: F,
) -> Result<Option<DirectColumnScanMetrics>>
where
    R: ChunkReader + 'static,
    F: FnMut(&[i32], &[i64], &[i16], &[ByteArray]) -> Result<Option<()>>,
{
    let Some(column_indices) = parquet_column_indices_by_name(&reader, &columns) else {
        return Ok(None);
    };
    let [predicate_column, sum_column, group_column] = <[usize; 3]>::try_from(column_indices)
        .map_err(|_| {
            DodamError::UnsupportedSql("direct parquet column index shape mismatch".to_string())
        })?;
    let schema = reader.metadata().file_metadata().schema_descr();
    let predicate_required = schema.column(predicate_column).max_def_level() == 0;
    let sum_required = schema.column(sum_column).max_def_level() == 0;
    let mut metrics = DirectColumnScanMetrics {
        row_groups: row_groups.len(),
        ..DirectColumnScanMetrics::default()
    };
    for &row_group_index in row_groups {
        let row_group = reader.get_row_group(row_group_index)?;
        let mut predicate_reader = match row_group.get_column_reader(predicate_column)? {
            ColumnReader::Int32ColumnReader(reader) => reader,
            _ => return Ok(None),
        };
        let mut sum_reader = match row_group.get_column_reader(sum_column)? {
            ColumnReader::Int64ColumnReader(reader) => reader,
            _ => return Ok(None),
        };
        let mut group_reader = match row_group.get_column_reader(group_column)? {
            ColumnReader::ByteArrayColumnReader(reader) => reader,
            _ => return Ok(None),
        };
        let mut predicate_values = Vec::<i32>::with_capacity(batch_size);
        let mut predicate_def_levels = Vec::<i16>::with_capacity(batch_size);
        let mut sum_values = Vec::<i64>::with_capacity(batch_size);
        let mut sum_def_levels = Vec::<i16>::with_capacity(batch_size);
        let mut group_values = Vec::<ByteArray>::with_capacity(batch_size);
        let mut group_def_levels = Vec::<i16>::with_capacity(batch_size);
        loop {
            predicate_values.clear();
            predicate_def_levels.clear();
            sum_values.clear();
            sum_def_levels.clear();
            group_values.clear();
            group_def_levels.clear();
            let read_started = Instant::now();
            let (records, predicate_value_count, _) = predicate_reader.read_records(
                batch_size,
                (!predicate_required).then_some(&mut predicate_def_levels),
                None,
                &mut predicate_values,
            )?;
            if records == 0 {
                metrics.add_read_nanos(elapsed_nanos(read_started));
                break;
            }
            let (sum_records, sum_value_count, _) = sum_reader.read_records(
                records,
                (!sum_required).then_some(&mut sum_def_levels),
                None,
                &mut sum_values,
            )?;
            let (group_records, _group_value_count, group_level_count) = group_reader
                .read_records(
                    records,
                    Some(&mut group_def_levels),
                    None,
                    &mut group_values,
                )?;
            metrics.add_read_nanos(elapsed_nanos(read_started));
            if sum_records != records
                || group_records != records
                || predicate_value_count != records
                || sum_value_count != records
                || (!predicate_required && !direct_all_present(false, &predicate_def_levels))
                || (!sum_required && !direct_all_present(false, &sum_def_levels))
                || group_level_count != records
            {
                return Ok(None);
            }
            metrics.batches += 1;
            metrics.rows = metrics.rows.saturating_add(records);
            let consume_started = Instant::now();
            if consume(
                &predicate_values,
                &sum_values,
                &group_def_levels,
                &group_values,
            )?
            .is_none()
            {
                return Ok(None);
            }
            metrics.add_consume_nanos(elapsed_nanos(consume_started));
        }
    }
    Ok(Some(metrics))
}

fn scan_parquet_i32_byte_array_columns_reader<R, F>(
    reader: SerializedFileReader<R>,
    batch_size: usize,
    row_groups: &[usize],
    columns: [&str; 2],
    mut consume: F,
) -> Result<Option<DirectColumnScanMetrics>>
where
    R: ChunkReader + 'static,
    F: FnMut(&[i32], &[i16], &[ByteArray]) -> Result<Option<()>>,
{
    let Some(column_indices) = parquet_column_indices_by_name(&reader, &columns) else {
        return Ok(None);
    };
    let [numeric_column, byte_array_column] =
        <[usize; 2]>::try_from(column_indices).map_err(|_| {
            DodamError::UnsupportedSql("direct parquet column index shape mismatch".to_string())
        })?;
    let schema = reader.metadata().file_metadata().schema_descr();
    let numeric_required = schema.column(numeric_column).max_def_level() == 0;
    let mut metrics = DirectColumnScanMetrics {
        row_groups: row_groups.len(),
        ..DirectColumnScanMetrics::default()
    };
    for &row_group_index in row_groups {
        let row_group = reader.get_row_group(row_group_index)?;
        let mut numeric_reader = match row_group.get_column_reader(numeric_column)? {
            ColumnReader::Int32ColumnReader(reader) => reader,
            _ => return Ok(None),
        };
        let mut byte_array_reader = match row_group.get_column_reader(byte_array_column)? {
            ColumnReader::ByteArrayColumnReader(reader) => reader,
            _ => return Ok(None),
        };
        let mut numeric_values = Vec::<i32>::with_capacity(batch_size);
        let mut numeric_def_levels = Vec::<i16>::with_capacity(batch_size);
        let mut byte_array_values = Vec::<ByteArray>::with_capacity(batch_size);
        let mut byte_array_def_levels = Vec::<i16>::with_capacity(batch_size);
        loop {
            numeric_values.clear();
            numeric_def_levels.clear();
            byte_array_values.clear();
            byte_array_def_levels.clear();
            let read_started = Instant::now();
            let (records, numeric_value_count, _) = numeric_reader.read_records(
                batch_size,
                (!numeric_required).then_some(&mut numeric_def_levels),
                None,
                &mut numeric_values,
            )?;
            if records == 0 {
                metrics.add_read_nanos(elapsed_nanos(read_started));
                break;
            }
            let (byte_array_records, _byte_array_value_count, byte_array_level_count) =
                byte_array_reader.read_records(
                    records,
                    Some(&mut byte_array_def_levels),
                    None,
                    &mut byte_array_values,
                )?;
            metrics.add_read_nanos(elapsed_nanos(read_started));
            if byte_array_records != records
                || numeric_value_count != records
                || (!numeric_required && !direct_all_present(false, &numeric_def_levels))
                || byte_array_level_count != records
            {
                return Ok(None);
            }
            metrics.batches += 1;
            metrics.rows = metrics.rows.saturating_add(records);
            let consume_started = Instant::now();
            if consume(&numeric_values, &byte_array_def_levels, &byte_array_values)?.is_none() {
                return Ok(None);
            }
            metrics.add_consume_nanos(elapsed_nanos(consume_started));
        }
    }
    Ok(Some(metrics))
}

fn scan_parquet_i32_i32_columns_reader<R, F>(
    reader: SerializedFileReader<R>,
    batch_size: usize,
    row_groups: &[usize],
    columns: [&str; 2],
    mut consume: F,
) -> Result<Option<DirectColumnScanMetrics>>
where
    R: ChunkReader + 'static,
    F: FnMut(&[i32], Option<&[i16]>, &[i32], Option<&[i16]>) -> Result<Option<()>>,
{
    let Some(column_indices) = parquet_column_indices_by_name(&reader, &columns) else {
        return Ok(None);
    };
    let [left_column, right_column] = <[usize; 2]>::try_from(column_indices).map_err(|_| {
        DodamError::UnsupportedSql("direct parquet column index shape mismatch".to_string())
    })?;
    let schema = reader.metadata().file_metadata().schema_descr();
    let left_required = schema.column(left_column).max_def_level() == 0;
    let right_required = schema.column(right_column).max_def_level() == 0;
    let mut metrics = DirectColumnScanMetrics {
        row_groups: row_groups.len(),
        ..DirectColumnScanMetrics::default()
    };
    let mut left_values = Vec::<i32>::with_capacity(batch_size);
    let mut right_values = Vec::<i32>::with_capacity(batch_size);
    let mut left_def_levels = Vec::<i16>::with_capacity(batch_size);
    let mut right_def_levels = Vec::<i16>::with_capacity(batch_size);
    let mut aligned_left_values = Vec::<i32>::with_capacity(batch_size);
    let mut aligned_right_values = Vec::<i32>::with_capacity(batch_size);
    for &row_group_index in row_groups {
        let row_group = reader.get_row_group(row_group_index)?;
        let mut left_reader = match row_group.get_column_reader(left_column)? {
            ColumnReader::Int32ColumnReader(reader) => reader,
            _ => return Ok(None),
        };
        let mut right_reader = match row_group.get_column_reader(right_column)? {
            ColumnReader::Int32ColumnReader(reader) => reader,
            _ => return Ok(None),
        };
        loop {
            left_values.clear();
            right_values.clear();
            left_def_levels.clear();
            right_def_levels.clear();
            aligned_left_values.clear();
            aligned_right_values.clear();
            let read_started = Instant::now();
            let (records, left_value_count, left_level_count) = left_reader.read_records(
                batch_size,
                (!left_required).then_some(&mut left_def_levels),
                None,
                &mut left_values,
            )?;
            if records == 0 {
                metrics.add_read_nanos(elapsed_nanos(read_started));
                break;
            }
            let (right_records, right_value_count, right_level_count) = right_reader.read_records(
                records,
                (!right_required).then_some(&mut right_def_levels),
                None,
                &mut right_values,
            )?;
            metrics.add_read_nanos(elapsed_nanos(read_started));
            if right_records != records
                || !direct_def_levels_match(left_level_count, records, left_required)
                || !direct_def_levels_match(right_level_count, records, right_required)
            {
                return Ok(None);
            }
            let left_slice = align_i32_records(
                records,
                left_required,
                &left_def_levels,
                &left_values,
                left_value_count,
                &mut aligned_left_values,
            )?;
            let right_slice = align_i32_records(
                records,
                right_required,
                &right_def_levels,
                &right_values,
                right_value_count,
                &mut aligned_right_values,
            )?;
            metrics.batches += 1;
            metrics.rows = metrics.rows.saturating_add(records);
            let consume_started = Instant::now();
            if consume(
                left_slice,
                (!left_required).then_some(left_def_levels.as_slice()),
                right_slice,
                (!right_required).then_some(right_def_levels.as_slice()),
            )?
            .is_none()
            {
                return Ok(None);
            }
            metrics.add_consume_nanos(elapsed_nanos(consume_started));
        }
    }
    Ok(Some(metrics))
}

fn align_i32_records<'a>(
    records: usize,
    required: bool,
    def_levels: &[i16],
    values: &'a [i32],
    value_count: usize,
    output: &'a mut Vec<i32>,
) -> Result<&'a [i32]> {
    if required || def_levels.is_empty() {
        if value_count != records || values.len() != records {
            return Err(DodamError::UnsupportedSql(
                "direct i32 record/value length mismatch".to_string(),
            ));
        }
        return Ok(values);
    }
    if def_levels.len() != records || value_count != values.len() {
        return Err(DodamError::UnsupportedSql(
            "direct nullable i32 level/value length mismatch".to_string(),
        ));
    }
    output.clear();
    output.reserve(records);
    let mut value_offset = 0usize;
    for level in def_levels {
        if *level == 0 {
            output.push(0);
        } else {
            let Some(value) = values.get(value_offset) else {
                return Err(DodamError::UnsupportedSql(
                    "direct nullable i32 value length mismatch".to_string(),
                ));
            };
            output.push(*value);
            value_offset += 1;
        }
    }
    if value_offset != values.len() {
        return Err(DodamError::UnsupportedSql(
            "direct nullable i32 unused values".to_string(),
        ));
    }
    Ok(output.as_slice())
}

fn scan_parquet_i32_byte_array_selected_by_i32_reader<R, P, F>(
    reader: SerializedFileReader<R>,
    row_groups: &[usize],
    columns: [&str; 2],
    predicate: P,
    mut consume: F,
) -> Result<Option<DirectColumnScanMetrics>>
where
    R: ChunkReader + 'static,
    P: Fn(i32) -> bool,
    F: FnMut(&[i32], &[i32], &[Bytes]) -> Result<Option<()>>,
{
    let Some(column_indices) = parquet_column_indices_by_name(&reader, &columns) else {
        return Ok(None);
    };
    let [numeric_column, byte_array_column] =
        <[usize; 2]>::try_from(column_indices).map_err(|_| {
            DodamError::UnsupportedSql("direct parquet column index shape mismatch".to_string())
        })?;
    let schema = reader.metadata().file_metadata().schema_descr();
    let numeric_required = schema.column(numeric_column).max_def_level() == 0;
    let mut metrics = DirectColumnScanMetrics {
        row_groups: row_groups.len(),
        ..DirectColumnScanMetrics::default()
    };
    let mut numeric_values = Vec::<i32>::new();
    let mut numeric_def_levels = Vec::<i16>::new();
    let mut selected_numbers = Vec::<i32>::new();
    let mut selected_runs = Vec::<(usize, usize)>::new();
    for &row_group_index in row_groups {
        let row_group = reader.get_row_group(row_group_index)?;
        let row_count = usize::try_from(row_group.metadata().num_rows()).map_err(|_| {
            DodamError::UnsupportedSql("row group row count out of range".to_string())
        })?;
        numeric_values.clear();
        numeric_def_levels.clear();
        selected_numbers.clear();
        selected_runs.clear();

        let read_started = Instant::now();
        let mut numeric_reader = match row_group.get_column_reader(numeric_column)? {
            ColumnReader::Int32ColumnReader(reader) => reader,
            _ => return Ok(None),
        };
        let (records, value_count, level_count) = numeric_reader.read_records(
            row_count,
            (!numeric_required).then_some(&mut numeric_def_levels),
            None,
            &mut numeric_values,
        )?;
        if records != row_count
            || value_count != row_count
            || !direct_def_levels_match(level_count, row_count, numeric_required)
            || !direct_all_present(numeric_required, &numeric_def_levels)
        {
            return Ok(None);
        }

        let mut builder = SelectionRunsBuilder::default();
        let mut run_start = None::<usize>;
        let mut run_len = 0usize;
        for (row, &number) in numeric_values.iter().enumerate() {
            if predicate(number) {
                selected_numbers.push(number);
                if run_start.is_some() {
                    run_len += 1;
                } else {
                    run_start = Some(row);
                    run_len = 1;
                }
            } else if let Some(start) = run_start.take() {
                builder.push_run(&mut selected_runs, start, run_len);
                run_len = 0;
            }
        }
        if let Some(start) = run_start {
            builder.push_run(&mut selected_runs, start, run_len);
        }
        let selected_rows = builder.selected_rows();
        metrics.rows = metrics.rows.saturating_add(row_count);
        metrics.selected_rows = metrics.selected_rows.saturating_add(selected_rows);
        metrics.selected_runs = metrics.selected_runs.saturating_add(selected_runs.len());
        if selected_rows != selected_numbers.len() {
            return Ok(None);
        }
        if selected_rows == 0 {
            metrics.add_read_nanos(elapsed_nanos(read_started));
            continue;
        }
        let Some((selected_dictionary_ids, dictionary)) =
            read_byte_array_dictionary_ids_selected_for_row_group(
                &*row_group,
                byte_array_column,
                &selected_runs,
                &[],
            )?
        else {
            return Ok(None);
        };
        metrics.add_read_nanos(elapsed_nanos(read_started));
        if selected_dictionary_ids.len() != selected_rows {
            return Ok(None);
        }
        metrics.batches += 1;
        let consume_started = Instant::now();
        if consume(&selected_numbers, &selected_dictionary_ids, &dictionary)?.is_none() {
            return Ok(None);
        }
        metrics.add_consume_nanos(elapsed_nanos(consume_started));
    }
    Ok(Some(metrics))
}

fn scan_parquet_i32_selected_by_byte_array_prefix_reader<R, F>(
    reader: SerializedFileReader<R>,
    row_groups: &[usize],
    columns: [&str; 2],
    prefix: &[u8],
    mut consume: F,
) -> Result<Option<DirectColumnScanMetrics>>
where
    R: ChunkReader + 'static,
    F: FnMut(&[i32], &[i32], &[Bytes]) -> Result<Option<()>>,
{
    let Some(column_indices) = parquet_column_indices_by_name(&reader, &columns) else {
        return Ok(None);
    };
    let [numeric_column, byte_array_column] =
        <[usize; 2]>::try_from(column_indices).map_err(|_| {
            DodamError::UnsupportedSql("direct parquet column index shape mismatch".to_string())
        })?;
    let schema = reader.metadata().file_metadata().schema_descr();
    let numeric_required = schema.column(numeric_column).max_def_level() == 0;
    let mut metrics = DirectColumnScanMetrics {
        row_groups: row_groups.len(),
        ..DirectColumnScanMetrics::default()
    };
    let mut selected_runs = Vec::<(usize, usize)>::new();
    let mut selected_numbers = Vec::<i32>::new();
    let mut numeric_def_levels = Vec::<i16>::new();
    for &row_group_index in row_groups {
        let row_group = reader.get_row_group(row_group_index)?;
        let row_count = usize::try_from(row_group.metadata().num_rows()).map_err(|_| {
            DodamError::UnsupportedSql("row group row count out of range".to_string())
        })?;
        selected_runs.clear();
        selected_numbers.clear();
        numeric_def_levels.clear();
        let read_started = Instant::now();
        let Some((dictionary_def_levels, dictionary_ids, dictionary)) =
            read_byte_array_dictionary_ids_for_row_group(&*row_group, byte_array_column)?
        else {
            return Ok(None);
        };
        if (!dictionary_def_levels.is_empty() && dictionary_def_levels.len() != row_count)
            || dictionary_ids.len() > row_count
        {
            return Ok(None);
        }
        let selected_dictionary_ids = dictionary
            .iter()
            .map(|value| value.as_ref().starts_with(prefix))
            .collect::<Vec<_>>();
        let mut builder = SelectionRunsBuilder::default();
        let mut dictionary_value_offset = 0usize;
        let mut run_start = None::<usize>;
        let mut run_len = 0usize;
        for row in 0..row_count {
            let present = dictionary_def_levels
                .get(row)
                .is_none_or(|level| *level != 0);
            let selected = if present {
                let Some(dictionary_id) = dictionary_ids.get(dictionary_value_offset) else {
                    return Ok(None);
                };
                dictionary_value_offset += 1;
                usize::try_from(*dictionary_id)
                    .ok()
                    .and_then(|id| selected_dictionary_ids.get(id))
                    .copied()
                    .unwrap_or(false)
            } else {
                false
            };
            if selected {
                if run_start.is_some() {
                    run_len += 1;
                } else {
                    run_start = Some(row);
                    run_len = 1;
                }
            } else if let Some(start) = run_start.take() {
                builder.push_run(&mut selected_runs, start, run_len);
                run_len = 0;
            }
        }
        if let Some(start) = run_start {
            builder.push_run(&mut selected_runs, start, run_len);
        }
        if dictionary_value_offset != dictionary_ids.len() {
            return Ok(None);
        }
        let selected_rows = builder.selected_rows();
        metrics.rows = metrics.rows.saturating_add(row_count);
        metrics.selected_rows = metrics.selected_rows.saturating_add(selected_rows);
        metrics.selected_runs = metrics.selected_runs.saturating_add(selected_runs.len());
        if selected_rows == 0 {
            metrics.add_read_nanos(elapsed_nanos(read_started));
            continue;
        }
        let mut numeric_reader = match row_group.get_column_reader(numeric_column)? {
            ColumnReader::Int32ColumnReader(reader) => reader,
            _ => return Ok(None),
        };
        let mut numeric_metrics = DirectPrimitiveColumnScanMetrics::default();
        if !read_i32_selected_runs(
            &mut numeric_reader,
            row_count,
            &selected_runs,
            numeric_required,
            &mut numeric_def_levels,
            &mut selected_numbers,
            &mut numeric_metrics,
        )? {
            return Ok(None);
        }
        let mut selected_dictionary_ids = Vec::<i32>::with_capacity(selected_rows);
        compact_selected_dictionary_ids(
            &dictionary_def_levels,
            &dictionary_ids,
            0,
            row_count,
            0,
            0,
            &selected_runs,
            &mut selected_dictionary_ids,
        )?;
        metrics.add_read_nanos(elapsed_nanos(read_started));
        if selected_numbers.len() != selected_rows || selected_dictionary_ids.len() != selected_rows
        {
            return Ok(None);
        }
        metrics.batches += 1;
        let consume_started = Instant::now();
        if consume(&selected_numbers, &selected_dictionary_ids, &dictionary)?.is_none() {
            return Ok(None);
        }
        metrics.add_consume_nanos(elapsed_nanos(consume_started));
    }
    Ok(Some(metrics))
}

fn scan_parquet_i64_byte_array_payload_columns_reader<R, F>(
    reader: SerializedFileReader<R>,
    batch_size: usize,
    row_groups: &[usize],
    columns: [&str; 2],
    mut consume: F,
) -> Result<Option<DirectColumnScanMetrics>>
where
    R: ChunkReader + 'static,
    F: for<'a> FnMut(&[i64], &mut DirectByteArrayPayloadReader<'a>) -> Result<Option<()>>,
{
    let Some(column_indices) = parquet_column_indices_by_name(&reader, &columns) else {
        return Ok(None);
    };
    let [key_column, payload_column] = <[usize; 2]>::try_from(column_indices).map_err(|_| {
        DodamError::UnsupportedSql("direct parquet column index shape mismatch".to_string())
    })?;
    let mut metrics = DirectColumnScanMetrics {
        row_groups: row_groups.len(),
        ..DirectColumnScanMetrics::default()
    };
    for &row_group_index in row_groups {
        let row_group = reader.get_row_group(row_group_index)?;
        let mut key_reader = match row_group.get_column_reader(key_column)? {
            ColumnReader::Int64ColumnReader(reader) => reader,
            _ => return Ok(None),
        };
        let mut payload_reader = match row_group.get_column_reader(payload_column)? {
            ColumnReader::ByteArrayColumnReader(reader) => reader,
            _ => return Ok(None),
        };
        let mut key_values = Vec::<i64>::with_capacity(batch_size);
        let mut key_def_levels = Vec::<i16>::with_capacity(batch_size);
        loop {
            key_values.clear();
            key_def_levels.clear();
            let read_started = Instant::now();
            let (key_records, key_value_count, _key_levels) = key_reader.read_records(
                batch_size,
                Some(&mut key_def_levels),
                None,
                &mut key_values,
            )?;
            metrics.add_read_nanos(elapsed_nanos(read_started));
            if key_records == 0 {
                break;
            }
            if key_value_count != key_records {
                return Ok(None);
            }
            metrics.batches += 1;
            metrics.rows = metrics.rows.saturating_add(key_records);
            let consume_started = Instant::now();
            let mut payload = DirectByteArrayPayloadReader::new(&mut payload_reader);
            if consume(&key_values, &mut payload)?.is_none() {
                return Ok(None);
            }
            let payload_read_nanos = payload.take_read_nanos();
            let consume_nanos = elapsed_nanos(consume_started);
            metrics.add_read_nanos(payload_read_nanos);
            metrics.add_consume_nanos(consume_nanos.saturating_sub(payload_read_nanos));
        }
    }
    Ok(Some(metrics))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn scan_parquet_primitive_columns_with_store<F>(
    path: &Path,
    batch_size: usize,
    row_groups: &[usize],
    columns: &[DirectPrimitiveColumnSpec<'_>],
    file_cache: Arc<ParquetFileCache>,
    store: &dyn ObjectStore,
    consume: F,
) -> Result<Option<DirectPrimitiveColumnScanMetrics>>
where
    F: for<'a> FnMut(&[RawColumnView<'a>]) -> Result<()>,
{
    scan_parquet_primitive_columns_with_store_impl(
        path, batch_size, row_groups, columns, file_cache, store, false, consume,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn scan_parquet_primitive_columns_with_store_page_reader<F>(
    path: &Path,
    batch_size: usize,
    row_groups: &[usize],
    columns: &[DirectPrimitiveColumnSpec<'_>],
    file_cache: Arc<ParquetFileCache>,
    store: &dyn ObjectStore,
    consume: F,
) -> Result<Option<DirectPrimitiveColumnScanMetrics>>
where
    F: for<'a> FnMut(&[RawColumnView<'a>]) -> Result<()>,
{
    scan_parquet_primitive_columns_with_store_impl(
        path, batch_size, row_groups, columns, file_cache, store, true, consume,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn scan_parquet_required_plain_primitive_in_list_desc_with_store<F>(
    path: &Path,
    batch_size: usize,
    row_groups: &[usize],
    columns: &[DirectPrimitiveColumnSpec<'_>],
    filter_index: usize,
    filter_i32_values: &[i32],
    filter_i64_values: &[i64],
    file_cache: Arc<ParquetFileCache>,
    store: &dyn ObjectStore,
    mut consume: F,
) -> Result<Option<DirectPrimitiveColumnScanMetrics>>
where
    F: FnMut(DirectOrderedPrimitiveBatch) -> Result<()>,
{
    if file_cache.enabled() {
        let reader = CachedParquetChunkReader::new(path, store, file_cache)?;
        let reader = SerializedFileReader::new(reader)?;
        return scan_parquet_required_plain_primitive_in_list_desc_reader(
            reader,
            batch_size,
            row_groups,
            columns,
            filter_index,
            filter_i32_values,
            filter_i64_values,
            &mut consume,
        );
    }
    let file = store.open(path)?;
    let reader = SerializedFileReader::new(file)?;
    scan_parquet_required_plain_primitive_in_list_desc_reader(
        reader,
        batch_size,
        row_groups,
        columns,
        filter_index,
        filter_i32_values,
        filter_i64_values,
        &mut consume,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn scan_parquet_required_plain_primitive_in_list_desc_selected_pages_with_store<F>(
    path: &Path,
    batch_size: usize,
    row_groups: &[usize],
    columns: &[DirectPrimitiveColumnSpec<'_>],
    filter_index: usize,
    filter_i32_values: &[i32],
    filter_i64_values: &[i64],
    file_cache: Arc<ParquetFileCache>,
    store: &dyn ObjectStore,
    mut consume: F,
) -> Result<Option<DirectPrimitiveColumnScanMetrics>>
where
    F: for<'a> FnMut(DirectSelectedPrimitivePageBatch<'a>) -> Result<()>,
{
    if file_cache.enabled() {
        let reader = CachedParquetChunkReader::new(path, store, file_cache)?;
        let reader = SerializedFileReader::new(reader)?;
        return scan_parquet_required_plain_primitive_in_list_desc_selected_pages_reader(
            reader,
            batch_size,
            row_groups,
            columns,
            filter_index,
            filter_i32_values,
            filter_i64_values,
            &mut consume,
        );
    }
    let file = store.open(path)?;
    let reader = SerializedFileReader::new(file)?;
    scan_parquet_required_plain_primitive_in_list_desc_selected_pages_reader(
        reader,
        batch_size,
        row_groups,
        columns,
        filter_index,
        filter_i32_values,
        filter_i64_values,
        &mut consume,
    )
}

#[allow(clippy::too_many_arguments)]
fn scan_parquet_primitive_columns_with_store_impl<F>(
    path: &Path,
    batch_size: usize,
    row_groups: &[usize],
    columns: &[DirectPrimitiveColumnSpec<'_>],
    file_cache: Arc<ParquetFileCache>,
    store: &dyn ObjectStore,
    force_page_reader: bool,
    consume: F,
) -> Result<Option<DirectPrimitiveColumnScanMetrics>>
where
    F: for<'a> FnMut(&[RawColumnView<'a>]) -> Result<()>,
{
    let mut consume = consume;
    if force_page_reader || direct_primitive_page_reader_enabled() {
        let page_result = if file_cache.enabled() {
            let reader = CachedParquetChunkReader::new(path, store, file_cache.clone())?;
            let reader = SerializedFileReader::new(reader)?;
            if let Some(metrics) = scan_parquet_primitive_columns_required_page_stream_reader(
                reader,
                batch_size,
                row_groups,
                columns,
                &mut consume,
            )? {
                return Ok(Some(metrics));
            }
            let reader = CachedParquetChunkReader::new(path, store, file_cache.clone())?;
            let reader = SerializedFileReader::new(reader)?;
            scan_parquet_primitive_columns_page_reader(
                reader,
                batch_size,
                row_groups,
                columns,
                &mut consume,
            )?
        } else {
            let file = store.open(path)?;
            let reader = SerializedFileReader::new(file)?;
            if let Some(metrics) = scan_parquet_primitive_columns_required_page_stream_reader(
                reader,
                batch_size,
                row_groups,
                columns,
                &mut consume,
            )? {
                return Ok(Some(metrics));
            }
            let file = store.open(path)?;
            let reader = SerializedFileReader::new(file)?;
            scan_parquet_primitive_columns_page_reader(
                reader,
                batch_size,
                row_groups,
                columns,
                &mut consume,
            )?
        };
        if page_result.is_some() {
            return Ok(page_result);
        }
    }
    if file_cache.enabled() {
        let reader = CachedParquetChunkReader::new(path, store, file_cache)?;
        let reader = SerializedFileReader::new(reader)?;
        return scan_parquet_primitive_columns_reader(
            reader, batch_size, row_groups, columns, consume,
        );
    }
    let file = store.open(path)?;
    let reader = SerializedFileReader::new(file)?;
    scan_parquet_primitive_columns_reader(reader, batch_size, row_groups, columns, consume)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn scan_parquet_required_primitive_count_sum_pages_with_store<F>(
    path: &Path,
    batch_size: usize,
    row_groups: &[usize],
    columns: &[DirectPrimitiveColumnSpec<'_>],
    file_cache: Arc<ParquetFileCache>,
    store: &dyn ObjectStore,
    consume: F,
) -> Result<Option<DirectPrimitiveColumnScanMetrics>>
where
    F: for<'a> FnMut(DirectPrimitiveCountSumPageBatch<'a>) -> Result<()>,
{
    if columns.len() != 2 {
        return Ok(None);
    }
    if file_cache.enabled() {
        let reader = CachedParquetChunkReader::new(path, store, file_cache)?;
        let reader = SerializedFileReader::new(reader)?;
        return scan_parquet_required_primitive_count_sum_pages_reader(
            reader, batch_size, row_groups, columns, consume,
        );
    }
    let file = store.open(path)?;
    let reader = SerializedFileReader::new(file)?;
    scan_parquet_required_primitive_count_sum_pages_reader(
        reader, batch_size, row_groups, columns, consume,
    )
}

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub(crate) fn scan_parquet_i32_i64_decimal_i32_selected_with_store<F>(
    path: &Path,
    batch_size: usize,
    row_groups: &[usize],
    columns: [&str; 4],
    decimal_precision: u8,
    decimal_scale: i8,
    decimal_min: Option<i64>,
    decimal_max: Option<i64>,
    date_min: Option<i32>,
    date_max: Option<i32>,
    file_cache: Arc<ParquetFileCache>,
    store: &dyn ObjectStore,
    mut consume: F,
) -> Result<Option<DirectPrimitiveColumnScanMetrics>>
where
    F: for<'a> FnMut(&[RawColumnView<'a>]) -> Result<()>,
{
    scan_parquet_i32_i64_decimal_i32_selected_typed_with_store(
        path,
        batch_size,
        row_groups,
        columns,
        decimal_precision,
        decimal_scale,
        decimal_min,
        decimal_max,
        date_min,
        date_max,
        file_cache,
        store,
        |batch| {
            let views = [
                RawColumnView::I32(batch.keys),
                RawColumnView::I64(batch.sums),
                RawColumnView::Decimal128I64 {
                    values: batch.decimals,
                    precision: decimal_precision,
                    scale: decimal_scale,
                },
                RawColumnView::Date32(batch.dates),
            ];
            consume(&views)
        },
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn scan_parquet_i32_i64_decimal_i32_selected_typed_with_store<F>(
    path: &Path,
    batch_size: usize,
    row_groups: &[usize],
    columns: [&str; 4],
    decimal_precision: u8,
    decimal_scale: i8,
    decimal_min: Option<i64>,
    decimal_max: Option<i64>,
    date_min: Option<i32>,
    date_max: Option<i32>,
    file_cache: Arc<ParquetFileCache>,
    store: &dyn ObjectStore,
    consume: F,
) -> Result<Option<DirectPrimitiveColumnScanMetrics>>
where
    F: for<'a> FnMut(DirectI32I64DecimalI32SelectedBatch<'a>) -> Result<()>,
{
    if file_cache.enabled() {
        let reader = CachedParquetChunkReader::new(path, store, file_cache)?;
        let reader = SerializedFileReader::new(reader)?;
        return scan_parquet_i32_i64_decimal_i32_selected_reader(
            reader,
            batch_size,
            row_groups,
            columns,
            decimal_precision,
            decimal_scale,
            decimal_min,
            decimal_max,
            date_min,
            date_max,
            consume,
        );
    }
    let file = store.open(path)?;
    let reader = SerializedFileReader::new(file)?;
    scan_parquet_i32_i64_decimal_i32_selected_reader(
        reader,
        batch_size,
        row_groups,
        columns,
        decimal_precision,
        decimal_scale,
        decimal_min,
        decimal_max,
        date_min,
        date_max,
        consume,
    )
}

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub(crate) fn scan_parquet_i32_i32_dictionary_i64_decimal_selected_with_store<F>(
    path: &Path,
    batch_size: usize,
    row_groups: &[usize],
    columns: [&str; 5],
    fallback: &[u8],
    decimal_min: Option<i64>,
    decimal_max: Option<i64>,
    file_cache: Arc<ParquetFileCache>,
    store: &dyn ObjectStore,
    mut consume: F,
) -> Result<Option<DirectPrimitiveColumnScanMetrics>>
where
    F: for<'a> FnMut(&[RawColumnView<'a>]) -> Result<()>,
{
    scan_parquet_i32_i32_dictionary_i64_decimal_selected_typed_with_store(
        path,
        batch_size,
        row_groups,
        columns,
        fallback,
        decimal_min,
        decimal_max,
        file_cache,
        store,
        |batch| {
            let views = [
                RawColumnView::I32(batch.first()),
                RawColumnView::Date32(batch.second()),
                RawColumnView::DictionaryI32 {
                    keys: batch.dictionary_ids(),
                    values: DictionaryStringValues::Bytes(batch.dictionary()),
                },
                RawColumnView::I64(batch.sums()),
            ];
            consume(&views)
        },
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn scan_parquet_i32_i32_dictionary_i64_decimal_selected_typed_with_store<F>(
    path: &Path,
    batch_size: usize,
    row_groups: &[usize],
    columns: [&str; 5],
    fallback: &[u8],
    decimal_min: Option<i64>,
    decimal_max: Option<i64>,
    file_cache: Arc<ParquetFileCache>,
    store: &dyn ObjectStore,
    consume: F,
) -> Result<Option<DirectPrimitiveColumnScanMetrics>>
where
    F: for<'a> FnMut(DirectI32I32DictionaryI64SelectedBatch<'a>) -> Result<()>,
{
    if file_cache.enabled() {
        let reader = CachedParquetChunkReader::new(path, store, file_cache)?;
        let reader = SerializedFileReader::new(reader)?;
        return scan_parquet_i32_i32_dictionary_i64_decimal_selected_reader(
            reader,
            batch_size,
            row_groups,
            columns,
            fallback,
            decimal_min,
            decimal_max,
            consume,
        );
    }
    let file = store.open(path)?;
    let reader = SerializedFileReader::new(file)?;
    scan_parquet_i32_i32_dictionary_i64_decimal_selected_reader(
        reader,
        batch_size,
        row_groups,
        columns,
        fallback,
        decimal_min,
        decimal_max,
        consume,
    )
}

#[allow(clippy::too_many_arguments)]
fn scan_parquet_i32_i32_dictionary_i64_decimal_selected_reader<R, F>(
    reader: SerializedFileReader<R>,
    batch_size: usize,
    row_groups: &[usize],
    columns: [&str; 5],
    fallback: &[u8],
    decimal_min: Option<i64>,
    decimal_max: Option<i64>,
    mut consume: F,
) -> Result<Option<DirectPrimitiveColumnScanMetrics>>
where
    R: ChunkReader + 'static,
    F: for<'a> FnMut(DirectI32I32DictionaryI64SelectedBatch<'a>) -> Result<()>,
{
    let Some(column_indices) = parquet_column_indices_by_name(&reader, &columns) else {
        return Ok(None);
    };
    let [
        first_column,
        second_column,
        dictionary_column,
        sum_column,
        decimal_column,
    ] = <[usize; 5]>::try_from(column_indices).map_err(|_| {
        DodamError::UnsupportedSql(
            "direct selected dictionary aggregate column shape mismatch".to_string(),
        )
    })?;
    let schema = reader.metadata().file_metadata().schema_descr();
    let first_required = schema.column(first_column).max_def_level() == 0;
    let second_required = schema.column(second_column).max_def_level() == 0;
    let sum_required = schema.column(sum_column).max_def_level() == 0;
    let decimal_required = schema.column(decimal_column).max_def_level() == 0;
    let mut metrics = DirectPrimitiveColumnScanMetrics {
        row_groups: row_groups.len(),
        column_read_nanos: vec![0; 5],
        ..DirectPrimitiveColumnScanMetrics::default()
    };
    if direct_dictionary_selected_page_decode_enabled() {
        return scan_parquet_i32_i32_dictionary_i64_decimal_selected_pages_reader(
            reader,
            batch_size,
            row_groups,
            (
                first_column,
                second_column,
                dictionary_column,
                sum_column,
                decimal_column,
            ),
            fallback,
            decimal_min,
            decimal_max,
            first_required,
            second_required,
            sum_required,
            decimal_required,
            metrics,
            consume,
        );
    }
    for &row_group_index in row_groups {
        let row_group = reader.get_row_group(row_group_index)?;
        let mut dictionary_payload: Option<(Vec<i16>, Vec<i32>, Vec<Bytes>, i32)> = None;

        let mut first_reader = match row_group.get_column_reader(first_column)? {
            ColumnReader::Int32ColumnReader(reader) => reader,
            _ => return Ok(None),
        };
        let mut second_reader = match row_group.get_column_reader(second_column)? {
            ColumnReader::Int32ColumnReader(reader) => reader,
            _ => return Ok(None),
        };
        let mut sum_reader = match row_group.get_column_reader(sum_column)? {
            ColumnReader::Int64ColumnReader(reader) => reader,
            _ => return Ok(None),
        };
        let mut decimal_reader = match row_group.get_column_reader(decimal_column)? {
            ColumnReader::Int64ColumnReader(reader) => reader,
            _ => return Ok(None),
        };
        let mut decimal_values = Vec::<i64>::with_capacity(batch_size);
        let mut decimal_def_levels = Vec::<i16>::with_capacity(batch_size);
        let mut first_values = Vec::<i32>::with_capacity(batch_size);
        let mut second_values = Vec::<i32>::with_capacity(batch_size);
        let mut sum_values = Vec::<i64>::with_capacity(batch_size);
        let mut first_def_levels = Vec::<i16>::with_capacity(batch_size);
        let mut second_def_levels = Vec::<i16>::with_capacity(batch_size);
        let mut sum_def_levels = Vec::<i16>::with_capacity(batch_size);
        let mut selected_runs = Vec::<(usize, usize)>::new();
        let mut selected_decimals = Vec::<i64>::with_capacity(batch_size);
        let mut ignored_dates = Vec::<i32>::with_capacity(batch_size);
        let mut selected_first = Vec::<i32>::with_capacity(batch_size);
        let mut selected_second = Vec::<i32>::with_capacity(batch_size);
        let mut selected_sums = Vec::<i64>::with_capacity(batch_size);
        let mut selected_dictionary_ids = Vec::<i32>::with_capacity(batch_size);
        let mut row_offset = 0usize;
        let mut dictionary_value_offset = 0usize;
        loop {
            decimal_values.clear();
            decimal_def_levels.clear();
            first_values.clear();
            second_values.clear();
            sum_values.clear();
            first_def_levels.clear();
            second_def_levels.clear();
            sum_def_levels.clear();
            selected_runs.clear();
            selected_decimals.clear();
            ignored_dates.clear();
            selected_first.clear();
            selected_second.clear();
            selected_sums.clear();
            selected_dictionary_ids.clear();

            let read_started = Instant::now();
            let decimal_started = Instant::now();
            let (records, decimal_value_count, decimal_level_count) = decimal_reader.read_records(
                batch_size,
                (!decimal_required).then_some(&mut decimal_def_levels),
                None,
                &mut decimal_values,
            )?;
            metrics.add_column_read_nanos(4, elapsed_nanos(decimal_started));
            if records == 0 {
                metrics.add_read_nanos(elapsed_nanos(read_started));
                break;
            }
            if decimal_value_count != records
                || !direct_def_levels_match(decimal_level_count, records, decimal_required)
                || !direct_all_present(decimal_required, &decimal_def_levels)
            {
                metrics.add_read_nanos(elapsed_nanos(read_started));
                return Ok(None);
            }
            ignored_dates.resize(records, 0);
            build_selected_runs(
                &decimal_values,
                &ignored_dates,
                decimal_min,
                decimal_max,
                None,
                None,
                &mut selected_runs,
                &mut selected_decimals,
                &mut Vec::new(),
            );
            metrics.selected_rows = metrics
                .selected_rows
                .saturating_add(selected_decimals.len());
            metrics.selected_runs = metrics.selected_runs.saturating_add(selected_runs.len());
            if row_offset + records > row_group.metadata().num_rows() as usize {
                metrics.add_read_nanos(elapsed_nanos(read_started));
                return Ok(None);
            }
            let use_selected_payload = direct_selection_payload_gate(
                records,
                selected_decimals.len(),
                selected_runs.len(),
            );
            if !use_selected_payload && !direct_dictionary_selected_full_payload_enabled() {
                metrics.add_read_nanos(elapsed_nanos(read_started));
                return Ok(None);
            }
            if dictionary_payload.is_none() {
                let dictionary_started = Instant::now();
                let Some((dictionary_def_levels, dictionary_ids, mut dictionary)) =
                    read_byte_array_dictionary_ids_for_row_group(&*row_group, dictionary_column)?
                else {
                    metrics.add_read_nanos(elapsed_nanos(read_started));
                    return Ok(None);
                };
                let fallback_id = dictionary_fallback_id(&mut dictionary, fallback)?;
                metrics.add_column_read_nanos(2, elapsed_nanos(dictionary_started));
                dictionary_payload = Some((
                    dictionary_def_levels,
                    dictionary_ids,
                    dictionary,
                    fallback_id,
                ));
            }

            let first_started = Instant::now();
            if use_selected_payload {
                if !read_i32_selected_runs(
                    &mut first_reader,
                    records,
                    &selected_runs,
                    first_required,
                    &mut first_def_levels,
                    &mut selected_first,
                    &mut metrics,
                )? {
                    metrics.add_read_nanos(elapsed_nanos(read_started));
                    return Ok(None);
                }
            } else {
                let (read_records, value_count, level_count) = first_reader.read_records(
                    records,
                    (!first_required).then_some(&mut first_def_levels),
                    None,
                    &mut first_values,
                )?;
                if read_records != records
                    || value_count != records
                    || !direct_def_levels_match(level_count, records, first_required)
                    || !direct_all_present(first_required, &first_def_levels)
                {
                    metrics.add_read_nanos(elapsed_nanos(read_started));
                    return Ok(None);
                }
            }
            metrics.add_column_read_nanos(0, elapsed_nanos(first_started));

            let second_started = Instant::now();
            if use_selected_payload {
                if !read_i32_selected_runs(
                    &mut second_reader,
                    records,
                    &selected_runs,
                    second_required,
                    &mut second_def_levels,
                    &mut selected_second,
                    &mut metrics,
                )? {
                    metrics.add_read_nanos(elapsed_nanos(read_started));
                    return Ok(None);
                }
            } else {
                let (read_records, value_count, level_count) = second_reader.read_records(
                    records,
                    (!second_required).then_some(&mut second_def_levels),
                    None,
                    &mut second_values,
                )?;
                if read_records != records
                    || value_count != records
                    || !direct_def_levels_match(level_count, records, second_required)
                    || !direct_all_present(second_required, &second_def_levels)
                {
                    metrics.add_read_nanos(elapsed_nanos(read_started));
                    return Ok(None);
                }
            }
            metrics.add_column_read_nanos(1, elapsed_nanos(second_started));

            let sum_started = Instant::now();
            if use_selected_payload {
                if !read_i64_selected_runs(
                    &mut sum_reader,
                    records,
                    &selected_runs,
                    sum_required,
                    &mut sum_def_levels,
                    &mut selected_sums,
                    &mut metrics,
                )? {
                    metrics.add_read_nanos(elapsed_nanos(read_started));
                    return Ok(None);
                }
                metrics.selected_payload_batches += 1;
            } else {
                let (read_records, value_count, level_count) = sum_reader.read_records(
                    records,
                    (!sum_required).then_some(&mut sum_def_levels),
                    None,
                    &mut sum_values,
                )?;
                if read_records != records
                    || value_count != records
                    || !direct_def_levels_match(level_count, records, sum_required)
                    || !direct_all_present(sum_required, &sum_def_levels)
                {
                    metrics.add_read_nanos(elapsed_nanos(read_started));
                    return Ok(None);
                }
                metrics.full_payload_batches += 1;
            }
            metrics.add_column_read_nanos(3, elapsed_nanos(sum_started));

            let Some((dictionary_def_levels, dictionary_ids, dictionary, fallback_id)) =
                dictionary_payload.as_ref()
            else {
                unreachable!("dictionary payload initialized before payload reads");
            };
            compact_selected_dictionary_ids(
                dictionary_def_levels,
                dictionary_ids,
                row_offset,
                records,
                dictionary_value_offset,
                *fallback_id,
                &selected_runs,
                &mut selected_dictionary_ids,
            )?;
            if !use_selected_payload {
                compact_selected_i32(&first_values, &selected_runs, &mut selected_first);
                compact_selected_i32(&second_values, &selected_runs, &mut selected_second);
                compact_selected_i64(&sum_values, &selected_runs, &mut selected_sums);
            }
            metrics.add_read_nanos(elapsed_nanos(read_started));
            metrics.batches += 1;
            metrics.rows = metrics.rows.saturating_add(records);
            if selected_sums.is_empty() {
                row_offset += records;
                dictionary_value_offset = advance_dictionary_value_offset(
                    &dictionary_def_levels,
                    row_offset - records,
                    records,
                    dictionary_value_offset,
                );
                continue;
            }
            let consume_started = Instant::now();
            consume(DirectI32I32DictionaryI64SelectedBatch::compact(
                &selected_first,
                &selected_second,
                &selected_dictionary_ids,
                dictionary,
                &selected_sums,
            ))?;
            metrics.add_consume_nanos(elapsed_nanos(consume_started));
            row_offset += records;
            dictionary_value_offset = advance_dictionary_value_offset(
                &dictionary_def_levels,
                row_offset - records,
                records,
                dictionary_value_offset,
            );
        }
    }
    Ok(Some(metrics))
}

#[allow(clippy::too_many_arguments)]
fn scan_parquet_i32_i32_dictionary_i64_decimal_selected_pages_reader<R, F>(
    reader: SerializedFileReader<R>,
    batch_size: usize,
    row_groups: &[usize],
    columns: (usize, usize, usize, usize, usize),
    fallback: &[u8],
    decimal_min: Option<i64>,
    decimal_max: Option<i64>,
    first_required: bool,
    second_required: bool,
    sum_required: bool,
    decimal_required: bool,
    mut metrics: DirectPrimitiveColumnScanMetrics,
    mut consume: F,
) -> Result<Option<DirectPrimitiveColumnScanMetrics>>
where
    R: ChunkReader + 'static,
    F: for<'a> FnMut(DirectI32I32DictionaryI64SelectedBatch<'a>) -> Result<()>,
{
    let (first_column, second_column, dictionary_column, sum_column, decimal_column) = columns;
    for &row_group_index in row_groups {
        let row_group = reader.get_row_group(row_group_index)?;
        let row_count = usize::try_from(row_group.metadata().num_rows()).map_err(|_| {
            DodamError::UnsupportedSql("row group row count out of range".to_string())
        })?;
        let mut decimal_values = Vec::<i64>::with_capacity(batch_size);
        let mut decimal_def_levels = Vec::<i16>::with_capacity(batch_size);
        let mut selected_runs = Vec::<(usize, usize)>::new();
        let mut selected_runs_builder = SelectionRunsBuilder::default();
        let predicate_started = Instant::now();
        let row_offset = if direct_dictionary_decimal_selected_runs_enabled() {
            let decimal_started = Instant::now();
            if let Some(selected_rows) = build_i64_dictionary_selected_runs_for_row_group(
                &*row_group,
                decimal_column,
                row_count,
                decimal_min,
                decimal_max,
                &mut selected_runs,
                &mut selected_runs_builder,
            )? {
                metrics.add_column_read_nanos(4, elapsed_nanos(decimal_started));
                debug_assert_eq!(selected_rows, selected_runs_builder.selected_rows());
                row_count
            } else {
                selected_runs.clear();
                selected_runs_builder = SelectionRunsBuilder::default();
                let mut decimal_reader = match row_group.get_column_reader(decimal_column)? {
                    ColumnReader::Int64ColumnReader(reader) => reader,
                    _ => return Ok(None),
                };
                let mut row_offset = 0usize;
                loop {
                    decimal_values.clear();
                    decimal_def_levels.clear();
                    let read_started = Instant::now();
                    let (records, value_count, level_count) = decimal_reader.read_records(
                        batch_size,
                        (!decimal_required).then_some(&mut decimal_def_levels),
                        None,
                        &mut decimal_values,
                    )?;
                    metrics.add_column_read_nanos(4, elapsed_nanos(read_started));
                    if records == 0 {
                        break;
                    }
                    if value_count != records
                        || !direct_def_levels_match(level_count, records, decimal_required)
                        || !direct_all_present(decimal_required, &decimal_def_levels)
                    {
                        return Ok(None);
                    }
                    append_decimal_selected_runs(
                        &decimal_values,
                        row_offset,
                        decimal_min,
                        decimal_max,
                        &mut selected_runs,
                        &mut selected_runs_builder,
                    );
                    row_offset += records;
                }
                row_offset
            }
        } else {
            let mut decimal_reader = match row_group.get_column_reader(decimal_column)? {
                ColumnReader::Int64ColumnReader(reader) => reader,
                _ => return Ok(None),
            };
            let mut row_offset = 0usize;
            loop {
                decimal_values.clear();
                decimal_def_levels.clear();
                let read_started = Instant::now();
                let (records, value_count, level_count) = decimal_reader.read_records(
                    batch_size,
                    (!decimal_required).then_some(&mut decimal_def_levels),
                    None,
                    &mut decimal_values,
                )?;
                metrics.add_column_read_nanos(4, elapsed_nanos(read_started));
                if records == 0 {
                    break;
                }
                if value_count != records
                    || !direct_def_levels_match(level_count, records, decimal_required)
                    || !direct_all_present(decimal_required, &decimal_def_levels)
                {
                    return Ok(None);
                }
                append_decimal_selected_runs(
                    &decimal_values,
                    row_offset,
                    decimal_min,
                    decimal_max,
                    &mut selected_runs,
                    &mut selected_runs_builder,
                );
                row_offset += records;
            }
            row_offset
        };
        metrics.add_selected_predicate_nanos(elapsed_nanos(predicate_started));
        if row_offset != row_count {
            return Ok(None);
        }
        let selected_rows = selected_runs_builder.selected_rows();
        metrics.rows = metrics.rows.saturating_add(row_count);
        metrics.selected_rows = metrics.selected_rows.saturating_add(selected_rows);
        metrics.selected_runs = metrics.selected_runs.saturating_add(selected_runs.len());
        if !direct_selection_payload_gate(row_count, selected_rows, selected_runs.len()) {
            return Ok(None);
        }
        if selected_rows == 0 {
            continue;
        }

        let mut first_reader = match row_group.get_column_reader(first_column)? {
            ColumnReader::Int32ColumnReader(reader) => reader,
            _ => return Ok(None),
        };
        let mut second_reader = match row_group.get_column_reader(second_column)? {
            ColumnReader::Int32ColumnReader(reader) => reader,
            _ => return Ok(None),
        };
        let mut sum_reader = match row_group.get_column_reader(sum_column)? {
            ColumnReader::Int64ColumnReader(reader) => reader,
            _ => return Ok(None),
        };
        let mut selected_first = Vec::<i32>::with_capacity(selected_rows);
        let mut selected_second = Vec::<i32>::with_capacity(selected_rows);
        let mut selected_sums = Vec::<i64>::with_capacity(selected_rows);
        let mut full_first = Vec::<i32>::with_capacity(row_count);
        let mut full_second = Vec::<i32>::with_capacity(row_count);
        let mut full_sums = Vec::<i64>::with_capacity(row_count);
        let mut scratch_i32 = Vec::<i32>::new();
        let mut scratch_i64 = Vec::<i64>::new();
        let mut first_def_levels = Vec::<i16>::new();
        let mut second_def_levels = Vec::<i16>::new();
        let mut sum_def_levels = Vec::<i16>::new();
        let mut masked_full_payload_ready = false;
        let mut streamed_sum_to_consumer = false;
        let mut sum_chunk_ranges = Vec::<SelectedI64ChunkRange>::new();
        let mut sum_chunk_values = Vec::<i64>::with_capacity(selected_rows);

        let read_started = Instant::now();
        let payload_started = Instant::now();
        if direct_dictionary_selected_primitive_page_slice_enabled() {
            let first_started = Instant::now();
            if read_i32_plain_page_selected_runs(
                &*row_group,
                first_column,
                row_count,
                &selected_runs,
                &mut selected_first,
                &mut metrics,
            )?
            .is_none()
            {
                if !read_i32_selected_runs(
                    &mut first_reader,
                    row_count,
                    &selected_runs,
                    first_required,
                    &mut first_def_levels,
                    &mut selected_first,
                    &mut metrics,
                )? {
                    return Ok(None);
                }
            }
            metrics.add_column_read_nanos(0, elapsed_nanos(first_started));

            let second_started = Instant::now();
            if read_i32_plain_page_selected_runs(
                &*row_group,
                second_column,
                row_count,
                &selected_runs,
                &mut selected_second,
                &mut metrics,
            )?
            .is_none()
            {
                if !read_i32_selected_runs(
                    &mut second_reader,
                    row_count,
                    &selected_runs,
                    second_required,
                    &mut second_def_levels,
                    &mut selected_second,
                    &mut metrics,
                )? {
                    return Ok(None);
                }
            }
            metrics.add_column_read_nanos(1, elapsed_nanos(second_started));

            let sum_started = Instant::now();
            let i64_page_result = if direct_fused_selected_i64_page_decoder_enabled() {
                selected_i64_decoder::read_plain_i64_selected_runs(
                    &*row_group,
                    sum_column,
                    row_count,
                    &selected_runs,
                    &mut selected_sums,
                    &mut metrics,
                )?
            } else {
                read_i64_plain_page_selected_runs(
                    &*row_group,
                    sum_column,
                    row_count,
                    &selected_runs,
                    &mut selected_sums,
                    &mut metrics,
                )?
            };
            if i64_page_result.is_none() {
                if !read_i64_selected_runs(
                    &mut sum_reader,
                    row_count,
                    &selected_runs,
                    sum_required,
                    &mut sum_def_levels,
                    &mut selected_sums,
                    &mut metrics,
                )? {
                    return Ok(None);
                }
            }
            metrics.selected_payload_batches += 1;
            metrics.add_column_read_nanos(3, elapsed_nanos(sum_started));
        } else if direct_dictionary_selected_full_primitive_payload_enabled() {
            let first_started = Instant::now();
            let (read_records, value_count, level_count) = first_reader.read_records(
                row_count,
                (!first_required).then_some(&mut first_def_levels),
                None,
                &mut full_first,
            )?;
            if read_records != row_count
                || value_count != row_count
                || !direct_def_levels_match(level_count, row_count, first_required)
                || !direct_all_present(first_required, &first_def_levels)
            {
                return Ok(None);
            }
            let masked_full_payload = direct_dictionary_selected_masked_full_payload_enabled();
            if !masked_full_payload {
                compact_selected_i32(&full_first, &selected_runs, &mut selected_first);
            }
            metrics.add_column_read_nanos(0, elapsed_nanos(first_started));

            let second_started = Instant::now();
            let (read_records, value_count, level_count) = second_reader.read_records(
                row_count,
                (!second_required).then_some(&mut second_def_levels),
                None,
                &mut full_second,
            )?;
            if read_records != row_count
                || value_count != row_count
                || !direct_def_levels_match(level_count, row_count, second_required)
                || !direct_all_present(second_required, &second_def_levels)
            {
                return Ok(None);
            }
            if !masked_full_payload {
                compact_selected_i32(&full_second, &selected_runs, &mut selected_second);
            }
            metrics.add_column_read_nanos(1, elapsed_nanos(second_started));

            let sum_started = Instant::now();
            let (read_records, value_count, level_count) = sum_reader.read_records(
                row_count,
                (!sum_required).then_some(&mut sum_def_levels),
                None,
                &mut full_sums,
            )?;
            if read_records != row_count
                || value_count != row_count
                || !direct_def_levels_match(level_count, row_count, sum_required)
                || !direct_all_present(sum_required, &sum_def_levels)
            {
                return Ok(None);
            }
            if masked_full_payload {
                masked_full_payload_ready = true;
            } else {
                compact_selected_i64(&full_sums, &selected_runs, &mut selected_sums);
            }
            metrics.full_payload_batches += 1;
            metrics.add_column_read_nanos(3, elapsed_nanos(sum_started));
        } else if direct_dictionary_selected_window_payload_enabled() {
            let windows = coalesce_selected_runs_to_windows(&selected_runs);
            let first_started = Instant::now();
            if !read_i32_selected_windows(
                &mut first_reader,
                row_count,
                &selected_runs,
                &windows,
                first_required,
                &mut first_def_levels,
                &mut scratch_i32,
                &mut selected_first,
                &mut metrics,
            )? {
                return Ok(None);
            }
            metrics.add_column_read_nanos(0, elapsed_nanos(first_started));

            let second_started = Instant::now();
            if !read_i32_selected_windows(
                &mut second_reader,
                row_count,
                &selected_runs,
                &windows,
                second_required,
                &mut second_def_levels,
                &mut scratch_i32,
                &mut selected_second,
                &mut metrics,
            )? {
                return Ok(None);
            }
            metrics.add_column_read_nanos(1, elapsed_nanos(second_started));

            let sum_started = Instant::now();
            if !read_i64_selected_windows(
                &mut sum_reader,
                row_count,
                &selected_runs,
                &windows,
                sum_required,
                &mut sum_def_levels,
                &mut scratch_i64,
                &mut selected_sums,
                &mut metrics,
            )? {
                return Ok(None);
            }
            metrics.selected_payload_batches += 1;
            metrics.add_column_read_nanos(3, elapsed_nanos(sum_started));
        } else {
            let first_started = Instant::now();
            if !read_i32_selected_runs(
                &mut first_reader,
                row_count,
                &selected_runs,
                first_required,
                &mut first_def_levels,
                &mut selected_first,
                &mut metrics,
            )? {
                return Ok(None);
            }
            metrics.add_column_read_nanos(0, elapsed_nanos(first_started));
            let second_started = Instant::now();
            if !read_i32_selected_runs(
                &mut second_reader,
                row_count,
                &selected_runs,
                second_required,
                &mut second_def_levels,
                &mut selected_second,
                &mut metrics,
            )? {
                return Ok(None);
            }
            metrics.add_column_read_nanos(1, elapsed_nanos(second_started));
            let sum_started = Instant::now();
            if direct_fused_selected_i64_aggregate_sink_enabled() {
                let dictionary_started = Instant::now();
                let Some((selected_dictionary_ids, dictionary)) =
                    read_byte_array_dictionary_ids_selected_for_row_group(
                        &*row_group,
                        dictionary_column,
                        &selected_runs,
                        fallback,
                    )?
                else {
                    return Ok(None);
                };
                let dictionary_nanos = elapsed_nanos(dictionary_started);
                metrics.add_column_read_nanos(2, dictionary_nanos);
                metrics.add_selected_dictionary_nanos(dictionary_nanos);
                if selected_first.len() != selected_rows
                    || selected_second.len() != selected_rows
                    || selected_dictionary_ids.len() != selected_rows
                {
                    return Ok(None);
                }
                let Some(()) = selected_i64_decoder::read_plain_i64_selected_runs_sink(
                    &*row_group,
                    sum_column,
                    row_count,
                    &selected_runs,
                    |selected_offset, sums| {
                        let end = selected_offset.saturating_add(sums.len());
                        if end > selected_rows {
                            return Err(DodamError::UnsupportedSql(
                                "selected i64 aggregate sink offset out of range".to_string(),
                            ));
                        }
                        metrics.add_selected_read(sums.len());
                        let value_offset = sum_chunk_values.len();
                        sum_chunk_values.extend_from_slice(sums);
                        sum_chunk_ranges.push(SelectedI64ChunkRange {
                            selected_offset,
                            values_offset: value_offset,
                            len: sums.len(),
                        });
                        Ok(())
                    },
                )?
                else {
                    return Ok(None);
                };
                let consume_started = Instant::now();
                consume(DirectI32I32DictionaryI64SelectedBatch::SumChunkRanges {
                    first: &selected_first,
                    second: &selected_second,
                    dictionary_ids: &selected_dictionary_ids,
                    dictionary: &dictionary,
                    chunks: &sum_chunk_ranges,
                    sums: &sum_chunk_values,
                    selected_rows,
                })?;
                metrics.add_consume_nanos(elapsed_nanos(consume_started));
                streamed_sum_to_consumer = true;
            } else if direct_dictionary_selected_i64_window_payload_enabled() {
                let windows = coalesce_selected_runs_to_i64_windows(&selected_runs);
                if !read_i64_selected_windows(
                    &mut sum_reader,
                    row_count,
                    &selected_runs,
                    &windows,
                    sum_required,
                    &mut sum_def_levels,
                    &mut scratch_i64,
                    &mut selected_sums,
                    &mut metrics,
                )? {
                    return Ok(None);
                }
            } else {
                if !read_i64_selected_runs(
                    &mut sum_reader,
                    row_count,
                    &selected_runs,
                    sum_required,
                    &mut sum_def_levels,
                    &mut selected_sums,
                    &mut metrics,
                )? {
                    return Ok(None);
                }
            }
            metrics.selected_payload_batches += 1;
            metrics.add_column_read_nanos(3, elapsed_nanos(sum_started));
        }
        metrics.add_selected_payload_nanos(elapsed_nanos(payload_started));
        if streamed_sum_to_consumer {
            metrics.add_read_nanos(elapsed_nanos(read_started));
            metrics.batches += 1;
            continue;
        }
        if masked_full_payload_ready {
            let dictionary_started = Instant::now();
            let Some((dictionary_def_levels, dictionary_ids, dictionary)) =
                read_byte_array_dictionary_ids_for_row_group(&*row_group, dictionary_column)?
            else {
                return Ok(None);
            };
            let dictionary_nanos = elapsed_nanos(dictionary_started);
            metrics.add_column_read_nanos(2, dictionary_nanos);
            metrics.add_selected_dictionary_nanos(dictionary_nanos);
            if dictionary_ids.len() != row_count
                || (!dictionary_def_levels.is_empty()
                    && !direct_all_present(true, &dictionary_def_levels))
            {
                return Ok(None);
            }
            metrics.add_read_nanos(elapsed_nanos(read_started));
            metrics.batches += 1;
            let consume_started = Instant::now();
            consume(DirectI32I32DictionaryI64SelectedBatch::Masked {
                first: &full_first,
                second: &full_second,
                dictionary_ids: &dictionary_ids,
                dictionary: &dictionary,
                sums: &full_sums,
                selection: SelectionRuns::new(&selected_runs, selected_rows),
            })?;
            metrics.add_consume_nanos(elapsed_nanos(consume_started));
        } else {
            let dictionary_started = Instant::now();
            let Some((selected_dictionary_ids, dictionary)) =
                read_byte_array_dictionary_ids_selected_for_row_group(
                    &*row_group,
                    dictionary_column,
                    &selected_runs,
                    fallback,
                )?
            else {
                return Ok(None);
            };
            let dictionary_nanos = elapsed_nanos(dictionary_started);
            metrics.add_column_read_nanos(2, dictionary_nanos);
            metrics.add_selected_dictionary_nanos(dictionary_nanos);
            if selected_first.len() != selected_rows
                || selected_second.len() != selected_rows
                || selected_sums.len() != selected_rows
                || selected_dictionary_ids.len() != selected_rows
            {
                return Ok(None);
            }
            metrics.add_read_nanos(elapsed_nanos(read_started));
            metrics.batches += 1;
            let consume_started = Instant::now();
            consume(DirectI32I32DictionaryI64SelectedBatch::compact(
                &selected_first,
                &selected_second,
                &selected_dictionary_ids,
                &dictionary,
                &selected_sums,
            ))?;
            metrics.add_consume_nanos(elapsed_nanos(consume_started));
        }
    }
    Ok(Some(metrics))
}

#[allow(clippy::too_many_arguments)]
fn scan_parquet_i32_i64_decimal_i32_selected_reader<R, F>(
    reader: SerializedFileReader<R>,
    batch_size: usize,
    row_groups: &[usize],
    columns: [&str; 4],
    _decimal_precision: u8,
    _decimal_scale: i8,
    decimal_min: Option<i64>,
    decimal_max: Option<i64>,
    date_min: Option<i32>,
    date_max: Option<i32>,
    mut consume: F,
) -> Result<Option<DirectPrimitiveColumnScanMetrics>>
where
    R: ChunkReader + 'static,
    F: for<'a> FnMut(DirectI32I64DecimalI32SelectedBatch<'a>) -> Result<()>,
{
    let Some(column_indices) = parquet_column_indices_by_name(&reader, &columns) else {
        return Ok(None);
    };
    let [key_column, sum_column, decimal_column, date_column] =
        <[usize; 4]>::try_from(column_indices).map_err(|_| {
            DodamError::UnsupportedSql(
                "direct selected primitive column shape mismatch".to_string(),
            )
        })?;
    let schema = reader.metadata().file_metadata().schema_descr();
    let key_required = schema.column(key_column).max_def_level() == 0;
    let sum_required = schema.column(sum_column).max_def_level() == 0;
    let decimal_required = schema.column(decimal_column).max_def_level() == 0;
    let date_required = schema.column(date_column).max_def_level() == 0;
    let mut metrics = DirectPrimitiveColumnScanMetrics {
        row_groups: row_groups.len(),
        column_read_nanos: vec![0; 4],
        ..DirectPrimitiveColumnScanMetrics::default()
    };
    for &row_group_index in row_groups {
        let row_group = reader.get_row_group(row_group_index)?;
        let mut key_reader = match row_group.get_column_reader(key_column)? {
            ColumnReader::Int32ColumnReader(reader) => reader,
            _ => return Ok(None),
        };
        let mut sum_reader = match row_group.get_column_reader(sum_column)? {
            ColumnReader::Int64ColumnReader(reader) => reader,
            _ => return Ok(None),
        };
        let mut decimal_reader = match row_group.get_column_reader(decimal_column)? {
            ColumnReader::Int64ColumnReader(reader) => reader,
            _ => return Ok(None),
        };
        let mut date_reader = match row_group.get_column_reader(date_column)? {
            ColumnReader::Int32ColumnReader(reader) => reader,
            _ => return Ok(None),
        };
        let mut decimal_values = Vec::<i64>::with_capacity(batch_size);
        let mut date_values = Vec::<i32>::with_capacity(batch_size);
        let mut key_def_levels = Vec::<i16>::with_capacity(batch_size);
        let mut sum_def_levels = Vec::<i16>::with_capacity(batch_size);
        let mut decimal_def_levels = Vec::<i16>::with_capacity(batch_size);
        let mut date_def_levels = Vec::<i16>::with_capacity(batch_size);
        let mut selected_keys = Vec::<i32>::with_capacity(batch_size);
        let mut selected_sums = Vec::<i64>::with_capacity(batch_size);
        let mut selected_decimals = Vec::<i64>::with_capacity(batch_size);
        let mut selected_dates = Vec::<i32>::with_capacity(batch_size);
        let mut selected_runs = Vec::<(usize, usize)>::new();
        loop {
            decimal_values.clear();
            date_values.clear();
            key_def_levels.clear();
            sum_def_levels.clear();
            decimal_def_levels.clear();
            date_def_levels.clear();
            selected_keys.clear();
            selected_sums.clear();
            selected_decimals.clear();
            selected_dates.clear();
            selected_runs.clear();
            let read_started = Instant::now();
            let decimal_started = Instant::now();
            let (records, value_count, level_count) = decimal_reader.read_records(
                batch_size,
                (!decimal_required).then_some(&mut decimal_def_levels),
                None,
                &mut decimal_values,
            )?;
            metrics.add_column_read_nanos(2, elapsed_nanos(decimal_started));
            if records == 0 {
                metrics.add_read_nanos(elapsed_nanos(read_started));
                break;
            }
            if value_count != records
                || !direct_def_levels_match(level_count, records, decimal_required)
                || !direct_all_present(decimal_required, &decimal_def_levels)
            {
                metrics.add_read_nanos(elapsed_nanos(read_started));
                return Ok(None);
            }
            let date_started = Instant::now();
            let (date_records, date_value_count, date_level_count) = date_reader.read_records(
                records,
                (!date_required).then_some(&mut date_def_levels),
                None,
                &mut date_values,
            )?;
            metrics.add_column_read_nanos(3, elapsed_nanos(date_started));
            if date_records != records
                || date_value_count != records
                || !direct_def_levels_match(date_level_count, records, date_required)
                || !direct_all_present(date_required, &date_def_levels)
            {
                metrics.add_read_nanos(elapsed_nanos(read_started));
                return Ok(None);
            }
            build_selected_runs(
                &decimal_values,
                &date_values,
                decimal_min,
                decimal_max,
                date_min,
                date_max,
                &mut selected_runs,
                &mut selected_decimals,
                &mut selected_dates,
            );
            metrics.selected_rows = metrics
                .selected_rows
                .saturating_add(selected_decimals.len());
            metrics.selected_runs = metrics.selected_runs.saturating_add(selected_runs.len());
            let use_selected_payload = direct_selection_payload_gate(
                records,
                selected_decimals.len(),
                selected_runs.len(),
            );
            let key_started = Instant::now();
            if use_selected_payload {
                if !read_i32_selected_runs(
                    &mut key_reader,
                    records,
                    &selected_runs,
                    key_required,
                    &mut key_def_levels,
                    &mut selected_keys,
                    &mut metrics,
                )? {
                    metrics.add_read_nanos(elapsed_nanos(read_started));
                    return Ok(None);
                }
            } else {
                let (key_records, key_value_count, key_level_count) = key_reader.read_records(
                    records,
                    (!key_required).then_some(&mut key_def_levels),
                    None,
                    &mut selected_keys,
                )?;
                if key_records != records
                    || key_value_count != records
                    || !direct_def_levels_match(key_level_count, records, key_required)
                    || !direct_all_present(key_required, &key_def_levels)
                {
                    metrics.add_read_nanos(elapsed_nanos(read_started));
                    return Ok(None);
                }
            }
            metrics.add_column_read_nanos(0, elapsed_nanos(key_started));
            let sum_started = Instant::now();
            if use_selected_payload {
                if !read_i64_selected_runs(
                    &mut sum_reader,
                    records,
                    &selected_runs,
                    sum_required,
                    &mut sum_def_levels,
                    &mut selected_sums,
                    &mut metrics,
                )? {
                    metrics.add_read_nanos(elapsed_nanos(read_started));
                    return Ok(None);
                }
                metrics.selected_payload_batches += 1;
            } else {
                let (sum_records, sum_value_count, sum_level_count) = sum_reader.read_records(
                    records,
                    (!sum_required).then_some(&mut sum_def_levels),
                    None,
                    &mut selected_sums,
                )?;
                if sum_records != records
                    || sum_value_count != records
                    || !direct_def_levels_match(sum_level_count, records, sum_required)
                    || !direct_all_present(sum_required, &sum_def_levels)
                {
                    metrics.add_read_nanos(elapsed_nanos(read_started));
                    return Ok(None);
                }
                metrics.full_payload_batches += 1;
            }
            metrics.add_column_read_nanos(1, elapsed_nanos(sum_started));
            metrics.add_read_nanos(elapsed_nanos(read_started));
            metrics.batches += 1;
            metrics.rows = metrics.rows.saturating_add(records);
            if selected_keys.is_empty() {
                continue;
            }
            let consume_started = Instant::now();
            let (decimal_view, date_view) = if use_selected_payload {
                (selected_decimals.as_slice(), selected_dates.as_slice())
            } else {
                (decimal_values.as_slice(), date_values.as_slice())
            };
            consume(DirectI32I64DecimalI32SelectedBatch {
                keys: &selected_keys,
                sums: &selected_sums,
                decimals: decimal_view,
                dates: date_view,
                predicate_applied: use_selected_payload,
            })?;
            metrics.add_consume_nanos(elapsed_nanos(consume_started));
        }
    }
    Ok(Some(metrics))
}

enum DirectPrimitiveColumnReader {
    I64(ColumnReaderImpl<Int64Type>),
    I32(ColumnReaderImpl<Int32Type>),
}

enum DirectPrimitiveColumnValues {
    I64(Vec<i64>),
    I32(Vec<i32>),
    Decimal128 {
        values: Vec<i128>,
        raw_i64: Vec<i64>,
    },
}

impl DirectPrimitiveColumnValues {
    fn new(column_type: DirectPrimitiveColumnType, capacity: usize) -> Self {
        match column_type {
            DirectPrimitiveColumnType::I64 => Self::I64(Vec::with_capacity(capacity)),
            DirectPrimitiveColumnType::I32 | DirectPrimitiveColumnType::Date32 => {
                Self::I32(Vec::with_capacity(capacity))
            }
            DirectPrimitiveColumnType::Decimal128Int64 { .. } => Self::Decimal128 {
                values: Vec::with_capacity(capacity),
                raw_i64: Vec::with_capacity(capacity),
            },
            DirectPrimitiveColumnType::Decimal128Int64Raw { .. } => {
                Self::I64(Vec::with_capacity(capacity))
            }
        }
    }

    fn clear(&mut self) {
        match self {
            Self::I64(values) => values.clear(),
            Self::I32(values) => values.clear(),
            Self::Decimal128 { values, raw_i64 } => {
                values.clear();
                raw_i64.clear();
            }
        }
    }

    fn value_count(&self) -> usize {
        match self {
            Self::I64(values) => values.len(),
            Self::I32(values) => values.len(),
            Self::Decimal128 { values, .. } => values.len(),
        }
    }

    fn as_view<'a>(
        &'a self,
        column_type: DirectPrimitiveColumnType,
        required: bool,
        def_levels: &'a [i16],
    ) -> RawColumnView<'a> {
        let null_free = required || def_levels.iter().all(|level| *level != 0);
        match (self, column_type) {
            (Self::I64(values), DirectPrimitiveColumnType::I64) if null_free => {
                RawColumnView::I64(values)
            }
            (Self::I64(values), DirectPrimitiveColumnType::I64) => {
                RawColumnView::I64Nullable { values, def_levels }
            }
            (Self::I32(values), DirectPrimitiveColumnType::I32) if null_free => {
                RawColumnView::I32(values)
            }
            (Self::I32(values), DirectPrimitiveColumnType::I32) => {
                RawColumnView::I32Nullable { values, def_levels }
            }
            (Self::I32(values), DirectPrimitiveColumnType::Date32) => RawColumnView::Date32(values),
            (
                Self::Decimal128 { values, .. },
                DirectPrimitiveColumnType::Decimal128Int64 { precision, scale },
            ) => RawColumnView::Decimal128 {
                values,
                precision,
                scale,
            },
            (
                Self::I64(values),
                DirectPrimitiveColumnType::Decimal128Int64Raw { precision, scale },
            ) => RawColumnView::Decimal128I64 {
                values,
                precision,
                scale,
            },
            _ => unreachable!("direct primitive values match their column spec"),
        }
    }
}

fn read_direct_primitive_records(
    reader: &mut DirectPrimitiveColumnReader,
    values: &mut DirectPrimitiveColumnValues,
    records: usize,
    required: bool,
    def_levels: &mut Vec<i16>,
) -> Result<(usize, usize, usize)> {
    match (reader, values) {
        (DirectPrimitiveColumnReader::I64(reader), DirectPrimitiveColumnValues::I64(values)) => {
            if required {
                Ok(reader.read_records(records, None, None, values)?)
            } else {
                Ok(reader.read_records(records, Some(def_levels), None, values)?)
            }
        }
        (DirectPrimitiveColumnReader::I32(reader), DirectPrimitiveColumnValues::I32(values)) => {
            if required {
                Ok(reader.read_records(records, None, None, values)?)
            } else {
                Ok(reader.read_records(records, Some(def_levels), None, values)?)
            }
        }
        (
            DirectPrimitiveColumnReader::I64(reader),
            DirectPrimitiveColumnValues::Decimal128 { values, raw_i64 },
        ) => {
            raw_i64.clear();
            let result = if required {
                reader.read_records(records, None, None, raw_i64)?
            } else {
                reader.read_records(records, Some(def_levels), None, raw_i64)?
            };
            values.extend(raw_i64.iter().copied().map(i128::from));
            Ok(result)
        }
        _ => Err(DodamError::UnsupportedSql(
            "direct primitive column reader/value type mismatch".to_string(),
        )),
    }
}

fn scan_parquet_primitive_columns_reader<R, F>(
    reader: SerializedFileReader<R>,
    batch_size: usize,
    row_groups: &[usize],
    columns: &[DirectPrimitiveColumnSpec<'_>],
    mut consume: F,
) -> Result<Option<DirectPrimitiveColumnScanMetrics>>
where
    R: ChunkReader + 'static,
    F: for<'a> FnMut(&[RawColumnView<'a>]) -> Result<()>,
{
    let names: Vec<&str> = columns.iter().map(|column| column.name).collect();
    let Some(column_indices) = parquet_column_indices_by_name(&reader, &names) else {
        return Ok(None);
    };
    let skip_required_def_levels = direct_skip_required_def_levels_enabled();
    let schema = reader.metadata().file_metadata().schema_descr();
    let required_columns: Vec<bool> = column_indices
        .iter()
        .map(|&index| skip_required_def_levels && schema.column(index).max_def_level() == 0)
        .collect();
    let mut metrics = DirectPrimitiveColumnScanMetrics {
        row_groups: row_groups.len(),
        column_read_nanos: vec![0; columns.len()],
        ..DirectPrimitiveColumnScanMetrics::default()
    };
    for &row_group_index in row_groups {
        let row_group = reader.get_row_group(row_group_index)?;
        let mut readers = Vec::with_capacity(columns.len());
        for (column, &column_index) in columns.iter().zip(column_indices.iter()) {
            let reader = match (
                column.column_type,
                row_group.get_column_reader(column_index)?,
            ) {
                (DirectPrimitiveColumnType::I64, ColumnReader::Int64ColumnReader(reader)) => {
                    DirectPrimitiveColumnReader::I64(reader)
                }
                (DirectPrimitiveColumnType::I32, ColumnReader::Int32ColumnReader(reader)) => {
                    DirectPrimitiveColumnReader::I32(reader)
                }
                (DirectPrimitiveColumnType::Date32, ColumnReader::Int32ColumnReader(reader)) => {
                    DirectPrimitiveColumnReader::I32(reader)
                }
                (
                    DirectPrimitiveColumnType::Decimal128Int64 { .. }
                    | DirectPrimitiveColumnType::Decimal128Int64Raw { .. },
                    ColumnReader::Int64ColumnReader(reader),
                ) => DirectPrimitiveColumnReader::I64(reader),
                _ => return Ok(None),
            };
            readers.push(reader);
        }
        let mut values: Vec<DirectPrimitiveColumnValues> = columns
            .iter()
            .map(|column| DirectPrimitiveColumnValues::new(column.column_type, batch_size))
            .collect();
        let mut def_levels: Vec<Vec<i16>> = columns
            .iter()
            .map(|_| Vec::<i16>::with_capacity(batch_size))
            .collect();
        loop {
            for values in &mut values {
                values.clear();
            }
            for levels in &mut def_levels {
                levels.clear();
            }
            let read_started = Instant::now();
            let mut record_count = 0usize;
            for index in 0..readers.len() {
                let requested = if index == 0 { batch_size } else { record_count };
                let column_read_started = Instant::now();
                let (records, value_count, level_count) = read_direct_primitive_records(
                    &mut readers[index],
                    &mut values[index],
                    requested,
                    required_columns[index],
                    &mut def_levels[index],
                )?;
                metrics.add_column_read_nanos(index, elapsed_nanos(column_read_started));
                if index == 0 {
                    record_count = records;
                    if record_count == 0 {
                        break;
                    }
                } else if records != record_count {
                    metrics.add_read_nanos(elapsed_nanos(read_started));
                    return Ok(None);
                }
                if !direct_value_count_matches(value_count, record_count, required_columns[index])
                    || !direct_def_levels_match(level_count, record_count, required_columns[index])
                {
                    metrics.add_read_nanos(elapsed_nanos(read_started));
                    return Ok(None);
                }
            }
            if record_count == 0 {
                metrics.add_read_nanos(elapsed_nanos(read_started));
                break;
            }
            metrics.add_read_nanos(elapsed_nanos(read_started));
            metrics.batches += 1;
            metrics.rows = metrics.rows.saturating_add(record_count);
            let consume_started = Instant::now();
            let views = direct_primitive_views(&values, columns, &required_columns, &def_levels);
            consume(&views)?;
            metrics.add_consume_nanos(elapsed_nanos(consume_started));
        }
    }
    Ok(Some(metrics))
}

fn direct_primitive_views<'a>(
    values: &'a [DirectPrimitiveColumnValues],
    columns: &[DirectPrimitiveColumnSpec<'_>],
    required_columns: &[bool],
    def_levels: &'a [Vec<i16>],
) -> Vec<RawColumnView<'a>> {
    values
        .iter()
        .zip(columns.iter())
        .zip(required_columns.iter())
        .zip(def_levels.iter())
        .map(|(((values, column), required), def_levels)| {
            values.as_view(column.column_type, *required, def_levels)
        })
        .collect()
}

fn scan_parquet_primitive_columns_required_page_stream_reader<R, F>(
    reader: SerializedFileReader<R>,
    batch_size: usize,
    row_groups: &[usize],
    columns: &[DirectPrimitiveColumnSpec<'_>],
    consume: &mut F,
) -> Result<Option<DirectPrimitiveColumnScanMetrics>>
where
    R: ChunkReader + 'static,
    F: for<'a> FnMut(&[RawColumnView<'a>]) -> Result<()>,
{
    let names: Vec<&str> = columns.iter().map(|column| column.name).collect();
    let Some(column_indices) = parquet_column_indices_by_name(&reader, &names) else {
        return Ok(None);
    };
    let schema = reader.metadata().file_metadata().schema_descr();
    if column_indices.iter().any(|&index| {
        schema.column(index).max_rep_level() != 0 || schema.column(index).max_def_level() != 0
    }) {
        return Ok(None);
    }
    for (column, &column_index) in columns.iter().zip(column_indices.iter()) {
        if !direct_primitive_page_column_type_matches(
            schema.column(column_index).physical_type(),
            column.column_type,
        ) {
            return Ok(None);
        }
    }
    let mut metrics = DirectPrimitiveColumnScanMetrics {
        row_groups: row_groups.len(),
        column_read_nanos: vec![0; columns.len()],
        ..DirectPrimitiveColumnScanMetrics::default()
    };
    for &row_group_index in row_groups {
        let row_group = reader.get_row_group(row_group_index)?;
        let row_count = usize::try_from(row_group.metadata().num_rows()).map_err(|_| {
            DodamError::UnsupportedSql("row group row count out of range".to_string())
        })?;
        let mut cursors = Vec::with_capacity(columns.len());
        for (column, &column_index) in columns.iter().zip(column_indices.iter()) {
            let page_reader = row_group.get_column_page_reader(column_index)?;
            cursors.push(RequiredPlainPrimitivePageCursor::new(
                page_reader,
                column.column_type,
                0,
            ));
        }
        let mut rows_read = 0usize;
        while rows_read < row_count {
            let read_started = Instant::now();
            let mut records = batch_size.min(row_count - rows_read);
            for (index, cursor) in cursors.iter_mut().enumerate() {
                let column_started = Instant::now();
                let loaded = cursor.ensure_page()?;
                metrics.add_column_read_nanos(index, elapsed_nanos(column_started));
                if !loaded {
                    metrics.add_read_nanos(elapsed_nanos(read_started));
                    return Ok(None);
                }
                records = records.min(cursor.available_rows());
            }
            metrics.add_read_nanos(elapsed_nanos(read_started));
            if records == 0 {
                return Ok(None);
            }
            let views = cursors
                .iter()
                .map(|cursor| cursor.raw_view(records))
                .collect::<Result<Vec<_>>>()?;
            let consume_started = Instant::now();
            consume(&views)?;
            metrics.add_consume_nanos(elapsed_nanos(consume_started));
            for cursor in &mut cursors {
                cursor.advance(records);
            }
            metrics.batches += 1;
            metrics.rows = metrics.rows.saturating_add(records);
            rows_read += records;
        }
    }
    Ok(Some(metrics))
}

#[allow(clippy::too_many_arguments)]
fn scan_parquet_required_plain_primitive_in_list_desc_reader<R, F>(
    reader: SerializedFileReader<R>,
    batch_size: usize,
    row_groups: &[usize],
    columns: &[DirectPrimitiveColumnSpec<'_>],
    filter_index: usize,
    filter_i32_values: &[i32],
    filter_i64_values: &[i64],
    consume: &mut F,
) -> Result<Option<DirectPrimitiveColumnScanMetrics>>
where
    R: ChunkReader + 'static,
    F: FnMut(DirectOrderedPrimitiveBatch) -> Result<()>,
{
    if columns.is_empty() || filter_index >= columns.len() {
        return Ok(None);
    }
    if columns.iter().any(|column| {
        !matches!(
            column.column_type,
            DirectPrimitiveColumnType::I32 | DirectPrimitiveColumnType::I64
        )
    }) {
        return Ok(None);
    }
    let names = columns.iter().map(|column| column.name).collect::<Vec<_>>();
    let Some(column_indices) = parquet_column_indices_by_name(&reader, &names) else {
        return Ok(None);
    };
    let schema = reader.metadata().file_metadata().schema_descr();
    for (column, &column_index) in columns.iter().zip(column_indices.iter()) {
        let parquet_column = schema.column(column_index);
        if parquet_column.max_rep_level() != 0
            || parquet_column.max_def_level() > 1
            || !direct_primitive_page_column_type_matches(
                parquet_column.physical_type(),
                column.column_type,
            )
        {
            return Ok(None);
        }
    }

    let mut metrics = DirectPrimitiveColumnScanMetrics {
        row_groups: row_groups.len(),
        column_read_nanos: vec![0; columns.len()],
        ..DirectPrimitiveColumnScanMetrics::default()
    };
    for &row_group_index in row_groups {
        let row_group = reader.get_row_group(row_group_index)?;
        let row_count = usize::try_from(row_group.metadata().num_rows()).map_err(|_| {
            DodamError::UnsupportedSql("row group row count out of range".to_string())
        })?;
        let filter_dictionary = if matches!(
            columns[filter_index].column_type,
            DirectPrimitiveColumnType::I32
        ) {
            read_i32_dictionary_ids_for_row_group(&*row_group, column_indices[filter_index])?
        } else {
            None
        };
        let mut cursors = Vec::with_capacity(columns.len());
        for (index, (column, &column_index)) in
            columns.iter().zip(column_indices.iter()).enumerate()
        {
            if filter_dictionary.is_some() && index == filter_index {
                cursors.push(None);
                continue;
            }
            let page_reader = row_group.get_column_page_reader(column_index)?;
            cursors.push(Some(RequiredPlainPrimitivePageCursor::new(
                page_reader,
                column.column_type,
                schema.column(column_index).max_def_level(),
            )));
        }
        let mut rows_read = 0usize;
        while rows_read < row_count {
            let read_started = Instant::now();
            let mut records = batch_size.min(row_count - rows_read);
            for (index, cursor) in cursors.iter_mut().enumerate() {
                let Some(cursor) = cursor else {
                    continue;
                };
                let column_started = Instant::now();
                let loaded = cursor.ensure_page()?;
                metrics.add_column_read_nanos(index, elapsed_nanos(column_started));
                if !loaded {
                    metrics.add_read_nanos(elapsed_nanos(read_started));
                    return Ok(None);
                }
                records = records.min(cursor.available_rows());
            }
            metrics.add_read_nanos(elapsed_nanos(read_started));
            if records == 0 {
                return Ok(None);
            }
            if let Some((ids, _)) = filter_dictionary.as_ref()
                && rows_read + records > ids.len()
            {
                return Ok(None);
            }
            let consume_started = Instant::now();
            let mut output = columns
                .iter()
                .map(|column| match column.column_type {
                    DirectPrimitiveColumnType::I32 => {
                        DirectOrderedPrimitiveColumnValues::I32(Vec::with_capacity(records / 4 + 1))
                    }
                    DirectPrimitiveColumnType::I64 => {
                        DirectOrderedPrimitiveColumnValues::I64(Vec::with_capacity(records / 4 + 1))
                    }
                    _ => unreachable!("checked primitive ordered column type"),
                })
                .collect::<Vec<_>>();
            let mut selected_positions = Vec::with_capacity(records / 4 + 1);
            match columns[filter_index].column_type {
                DirectPrimitiveColumnType::I32 => {
                    if let Some((ids, dictionary)) = filter_dictionary.as_ref() {
                        build_i32_dictionary_selected_positions_desc(
                            ids,
                            dictionary,
                            rows_read,
                            records,
                            filter_i32_values,
                            &mut selected_positions,
                        )?;
                    } else {
                        let Some(cursor) = cursors[filter_index].as_ref() else {
                            return Ok(None);
                        };
                        let Some(bytes) = cursor.raw_bytes(records) else {
                            return Ok(None);
                        };
                        build_i32_plain_selected_positions_desc(
                            bytes,
                            records,
                            filter_i32_values,
                            &mut selected_positions,
                        )?;
                    }
                }
                DirectPrimitiveColumnType::I64 => {
                    let Some(cursor) = cursors[filter_index].as_ref() else {
                        return Ok(None);
                    };
                    let Some(bytes) = cursor.raw_bytes(records) else {
                        return Ok(None);
                    };
                    build_i64_plain_selected_positions_desc(
                        bytes,
                        records,
                        filter_i64_values,
                        &mut selected_positions,
                    )?;
                }
                _ => return Ok(None),
            }
            for (index, column) in columns.iter().enumerate() {
                match (&mut output[index], column.column_type) {
                    (
                        DirectOrderedPrimitiveColumnValues::I32(values),
                        DirectPrimitiveColumnType::I32,
                    ) => {
                        if let Some((ids, dictionary)) = filter_dictionary.as_ref()
                            && index == filter_index
                        {
                            gather_i32_dictionary_positions(
                                ids,
                                dictionary,
                                rows_read,
                                &selected_positions,
                                values,
                            )?;
                        } else {
                            let Some(cursor) = cursors[index].as_ref() else {
                                return Ok(None);
                            };
                            let Some(bytes) = cursor.raw_bytes(records) else {
                                return Ok(None);
                            };
                            gather_i32_plain_positions(
                                bytes,
                                records,
                                &selected_positions,
                                values,
                            )?;
                        }
                    }
                    (
                        DirectOrderedPrimitiveColumnValues::I64(values),
                        DirectPrimitiveColumnType::I64,
                    ) => {
                        let Some(cursor) = cursors[index].as_ref() else {
                            return Ok(None);
                        };
                        let Some(bytes) = cursor.raw_bytes(records) else {
                            return Ok(None);
                        };
                        gather_i64_plain_positions(bytes, records, &selected_positions, values)?;
                    }
                    _ => return Ok(None),
                }
            }
            let selected_rows = output
                .first()
                .map(DirectOrderedPrimitiveColumnValues::len)
                .unwrap_or(0);
            if selected_rows > 0 {
                metrics.selected_rows = metrics.selected_rows.saturating_add(selected_rows);
                consume(DirectOrderedPrimitiveBatch { columns: output })?;
            }
            metrics.add_consume_nanos(elapsed_nanos(consume_started));
            for cursor in &mut cursors {
                if let Some(cursor) = cursor {
                    cursor.advance(records);
                }
            }
            rows_read += records;
            metrics.batches += 1;
            metrics.rows = metrics.rows.saturating_add(records);
        }
    }
    Ok(Some(metrics))
}

#[allow(clippy::too_many_arguments)]
fn scan_parquet_required_plain_primitive_in_list_desc_selected_pages_reader<R, F>(
    reader: SerializedFileReader<R>,
    batch_size: usize,
    row_groups: &[usize],
    columns: &[DirectPrimitiveColumnSpec<'_>],
    filter_index: usize,
    filter_i32_values: &[i32],
    filter_i64_values: &[i64],
    consume: &mut F,
) -> Result<Option<DirectPrimitiveColumnScanMetrics>>
where
    R: ChunkReader + 'static,
    F: for<'a> FnMut(DirectSelectedPrimitivePageBatch<'a>) -> Result<()>,
{
    if columns.is_empty() || filter_index >= columns.len() {
        return Ok(None);
    }
    if columns.iter().any(|column| {
        !matches!(
            column.column_type,
            DirectPrimitiveColumnType::I32 | DirectPrimitiveColumnType::I64
        )
    }) {
        return Ok(None);
    }
    let names = columns.iter().map(|column| column.name).collect::<Vec<_>>();
    let Some(column_indices) = parquet_column_indices_by_name(&reader, &names) else {
        return Ok(None);
    };
    let schema = reader.metadata().file_metadata().schema_descr();
    for (column, &column_index) in columns.iter().zip(column_indices.iter()) {
        let parquet_column = schema.column(column_index);
        if parquet_column.max_rep_level() != 0
            || parquet_column.max_def_level() > 1
            || !direct_primitive_page_column_type_matches(
                parquet_column.physical_type(),
                column.column_type,
            )
        {
            return Ok(None);
        }
    }

    let mut metrics = DirectPrimitiveColumnScanMetrics {
        row_groups: row_groups.len(),
        column_read_nanos: vec![0; columns.len()],
        ..DirectPrimitiveColumnScanMetrics::default()
    };
    for &row_group_index in row_groups {
        let row_group = reader.get_row_group(row_group_index)?;
        let row_count = usize::try_from(row_group.metadata().num_rows()).map_err(|_| {
            DodamError::UnsupportedSql("row group row count out of range".to_string())
        })?;
        let filter_dictionary = if matches!(
            columns[filter_index].column_type,
            DirectPrimitiveColumnType::I32
        ) {
            read_i32_dictionary_ids_for_row_group(&*row_group, column_indices[filter_index])?
        } else {
            None
        };
        let mut cursors = Vec::with_capacity(columns.len());
        for (index, (column, &column_index)) in
            columns.iter().zip(column_indices.iter()).enumerate()
        {
            if filter_dictionary.is_some() && index == filter_index {
                cursors.push(None);
                continue;
            }
            let page_reader = row_group.get_column_page_reader(column_index)?;
            cursors.push(Some(RequiredPlainPrimitivePageCursor::new(
                page_reader,
                column.column_type,
                schema.column(column_index).max_def_level(),
            )));
        }
        let mut rows_read = 0usize;
        let mut selected_positions = Vec::with_capacity(batch_size / 4 + 1);
        while rows_read < row_count {
            let read_started = Instant::now();
            let mut records = batch_size.min(row_count - rows_read);
            for (index, cursor) in cursors.iter_mut().enumerate() {
                let Some(cursor) = cursor else {
                    continue;
                };
                let column_started = Instant::now();
                let loaded = cursor.ensure_page()?;
                metrics.add_column_read_nanos(index, elapsed_nanos(column_started));
                if !loaded {
                    metrics.add_read_nanos(elapsed_nanos(read_started));
                    return Ok(None);
                }
                records = records.min(cursor.available_rows());
            }
            metrics.add_read_nanos(elapsed_nanos(read_started));
            if records == 0 {
                return Ok(None);
            }
            if let Some((ids, _)) = filter_dictionary.as_ref()
                && rows_read + records > ids.len()
            {
                return Ok(None);
            }

            let consume_started = Instant::now();
            selected_positions.clear();
            match columns[filter_index].column_type {
                DirectPrimitiveColumnType::I32 => {
                    if let Some((ids, dictionary)) = filter_dictionary.as_ref() {
                        build_i32_dictionary_selected_positions_desc(
                            ids,
                            dictionary,
                            rows_read,
                            records,
                            filter_i32_values,
                            &mut selected_positions,
                        )?;
                    } else {
                        let Some(cursor) = cursors[filter_index].as_ref() else {
                            return Ok(None);
                        };
                        let Some(bytes) = cursor.raw_bytes(records) else {
                            return Ok(None);
                        };
                        build_i32_plain_selected_positions_desc(
                            bytes,
                            records,
                            filter_i32_values,
                            &mut selected_positions,
                        )?;
                    }
                }
                DirectPrimitiveColumnType::I64 => {
                    let Some(cursor) = cursors[filter_index].as_ref() else {
                        return Ok(None);
                    };
                    let Some(bytes) = cursor.raw_bytes(records) else {
                        return Ok(None);
                    };
                    build_i64_plain_selected_positions_desc(
                        bytes,
                        records,
                        filter_i64_values,
                        &mut selected_positions,
                    )?;
                }
                _ => return Ok(None),
            }
            if !selected_positions.is_empty() {
                let mut page_columns = Vec::with_capacity(columns.len());
                for (index, column) in columns.iter().enumerate() {
                    match column.column_type {
                        DirectPrimitiveColumnType::I32 => {
                            if let Some((ids, dictionary)) = filter_dictionary.as_ref()
                                && index == filter_index
                            {
                                page_columns.push(
                                    DirectSelectedPrimitiveColumnPageView::I32Dictionary {
                                        ids,
                                        dictionary,
                                        rows_read,
                                    },
                                );
                            } else {
                                let Some(cursor) = cursors[index].as_ref() else {
                                    return Ok(None);
                                };
                                let Some(bytes) = cursor.raw_bytes(records) else {
                                    return Ok(None);
                                };
                                page_columns.push(
                                    DirectSelectedPrimitiveColumnPageView::I32Plain {
                                        bytes,
                                        records,
                                    },
                                );
                            }
                        }
                        DirectPrimitiveColumnType::I64 => {
                            let Some(cursor) = cursors[index].as_ref() else {
                                return Ok(None);
                            };
                            let Some(bytes) = cursor.raw_bytes(records) else {
                                return Ok(None);
                            };
                            page_columns.push(DirectSelectedPrimitiveColumnPageView::I64Plain {
                                bytes,
                                records,
                            });
                        }
                        _ => return Ok(None),
                    }
                }
                metrics.selected_rows = metrics
                    .selected_rows
                    .saturating_add(selected_positions.len());
                consume(DirectSelectedPrimitivePageBatch {
                    columns: page_columns,
                    selected_positions: &selected_positions,
                })?;
            }
            metrics.add_consume_nanos(elapsed_nanos(consume_started));
            for cursor in &mut cursors {
                if let Some(cursor) = cursor {
                    cursor.advance(records);
                }
            }
            rows_read += records;
            metrics.batches += 1;
            metrics.rows = metrics.rows.saturating_add(records);
        }
    }
    Ok(Some(metrics))
}

fn build_i32_plain_selected_positions_desc(
    bytes: &[u8],
    records: usize,
    filter_values: &[i32],
    output: &mut Vec<usize>,
) -> Result<()> {
    if bytes.len() < records.saturating_mul(std::mem::size_of::<i32>()) {
        return Ok(());
    }
    for row in (0..records).rev() {
        let value = read_i32_le_unchecked(bytes, row);
        if small_i32_values_contains(filter_values, value) {
            output.push(row);
        }
    }
    Ok(())
}

fn build_i64_plain_selected_positions_desc(
    bytes: &[u8],
    records: usize,
    filter_values: &[i64],
    output: &mut Vec<usize>,
) -> Result<()> {
    if bytes.len() < records.saturating_mul(std::mem::size_of::<i64>()) {
        return Ok(());
    }
    for row in (0..records).rev() {
        let value = read_i64_le_unchecked(bytes, row);
        if small_i64_values_contains(filter_values, value) {
            output.push(row);
        }
    }
    Ok(())
}

fn build_i32_dictionary_selected_positions_desc(
    ids: &[i32],
    dictionary: &[i32],
    rows_read: usize,
    records: usize,
    filter_values: &[i32],
    output: &mut Vec<usize>,
) -> Result<()> {
    let end = rows_read.saturating_add(records);
    if end > ids.len() {
        return Ok(());
    }
    for row in (0..records).rev() {
        let id = ids[rows_read + row];
        let Ok(id) = usize::try_from(id) else {
            return Ok(());
        };
        let Some(value) = dictionary.get(id).copied() else {
            return Ok(());
        };
        if small_i32_values_contains(filter_values, value) {
            output.push(row);
        }
    }
    Ok(())
}

fn gather_i32_plain_positions(
    bytes: &[u8],
    records: usize,
    positions: &[usize],
    output: &mut Vec<i32>,
) -> Result<()> {
    if bytes.len() < records.saturating_mul(std::mem::size_of::<i32>()) {
        return Ok(());
    }
    output.reserve(positions.len());
    for &row in positions {
        output.push(read_i32_le_unchecked(bytes, row));
    }
    Ok(())
}

fn gather_i64_plain_positions(
    bytes: &[u8],
    records: usize,
    positions: &[usize],
    output: &mut Vec<i64>,
) -> Result<()> {
    if bytes.len() < records.saturating_mul(std::mem::size_of::<i64>()) {
        return Ok(());
    }
    output.reserve(positions.len());
    for &row in positions {
        output.push(read_i64_le_unchecked(bytes, row));
    }
    Ok(())
}

fn gather_i32_dictionary_positions(
    ids: &[i32],
    dictionary: &[i32],
    rows_read: usize,
    positions: &[usize],
    output: &mut Vec<i32>,
) -> Result<()> {
    output.reserve(positions.len());
    for &row in positions {
        let Some(id) = ids.get(rows_read + row).copied() else {
            return Ok(());
        };
        let Ok(id) = usize::try_from(id) else {
            return Ok(());
        };
        let Some(value) = dictionary.get(id).copied() else {
            return Ok(());
        };
        output.push(value);
    }
    Ok(())
}

fn read_i32_le_unchecked(bytes: &[u8], row: usize) -> i32 {
    let start = row * std::mem::size_of::<i32>();
    i32::from_le_bytes([
        bytes[start],
        bytes[start + 1],
        bytes[start + 2],
        bytes[start + 3],
    ])
}

fn read_i64_le_unchecked(bytes: &[u8], row: usize) -> i64 {
    let start = row * std::mem::size_of::<i64>();
    i64::from_le_bytes([
        bytes[start],
        bytes[start + 1],
        bytes[start + 2],
        bytes[start + 3],
        bytes[start + 4],
        bytes[start + 5],
        bytes[start + 6],
        bytes[start + 7],
    ])
}

fn small_i32_values_contains(values: &[i32], value: i32) -> bool {
    match values {
        [] => false,
        [a] => value == *a,
        [a, b] => value == *a || value == *b,
        [a, b, c] => value == *a || value == *b || value == *c,
        [a, b, c, d] => value == *a || value == *b || value == *c || value == *d,
        _ => values.contains(&value),
    }
}

fn small_i64_values_contains(values: &[i64], value: i64) -> bool {
    match values {
        [] => false,
        [a] => value == *a,
        [a, b] => value == *a || value == *b,
        [a, b, c] => value == *a || value == *b || value == *c,
        [a, b, c, d] => value == *a || value == *b || value == *c || value == *d,
        _ => values.contains(&value),
    }
}

fn scan_parquet_required_primitive_count_sum_pages_reader<R, F>(
    reader: SerializedFileReader<R>,
    batch_size: usize,
    row_groups: &[usize],
    columns: &[DirectPrimitiveColumnSpec<'_>],
    mut consume: F,
) -> Result<Option<DirectPrimitiveColumnScanMetrics>>
where
    R: ChunkReader + 'static,
    F: for<'a> FnMut(DirectPrimitiveCountSumPageBatch<'a>) -> Result<()>,
{
    let [key_spec, sum_spec] = columns else {
        return Ok(None);
    };
    if !matches!(
        (key_spec.column_type, sum_spec.column_type),
        (
            DirectPrimitiveColumnType::I32,
            DirectPrimitiveColumnType::I64
        ) | (
            DirectPrimitiveColumnType::I64,
            DirectPrimitiveColumnType::I64
        )
    ) {
        return Ok(None);
    }
    let names = [key_spec.name, sum_spec.name];
    let Some(column_indices) = parquet_column_indices_by_name(&reader, &names) else {
        return Ok(None);
    };
    let [key_column, sum_column] = <[usize; 2]>::try_from(column_indices).map_err(|_| {
        DodamError::UnsupportedSql("direct primitive count/sum column shape mismatch".to_string())
    })?;
    let schema = reader.metadata().file_metadata().schema_descr();
    let key_column_desc = schema.column(key_column);
    let sum_column_desc = schema.column(sum_column);
    if key_column_desc.max_rep_level() != 0
        || sum_column_desc.max_rep_level() != 0
        || sum_column_desc.max_def_level() != 0
        || !direct_primitive_page_column_type_matches(
            key_column_desc.physical_type(),
            key_spec.column_type,
        )
        || !direct_primitive_page_column_type_matches(
            sum_column_desc.physical_type(),
            sum_spec.column_type,
        )
    {
        return Ok(None);
    }
    let optional_all_present_key = optional_all_present_count_sum_page_cursor_enabled()
        && key_column_desc.max_def_level() > 0
        && key_column_desc.max_def_level() <= 1;
    let nullable_i32_key = !optional_all_present_key
        && nullable_count_sum_page_cursor_enabled()
        && matches!(key_spec.column_type, DirectPrimitiveColumnType::I32)
        && key_column_desc.max_def_level() > 0;
    if key_column_desc.max_def_level() > 0
        && !optional_all_present_key
        && !(nullable_i32_key && matches!(sum_spec.column_type, DirectPrimitiveColumnType::I64))
    {
        return Ok(None);
    }
    let mut metrics = DirectPrimitiveColumnScanMetrics {
        row_groups: row_groups.len(),
        column_read_nanos: vec![0; 2],
        ..DirectPrimitiveColumnScanMetrics::default()
    };
    for &row_group_index in row_groups {
        let row_group = reader.get_row_group(row_group_index)?;
        let row_count = usize::try_from(row_group.metadata().num_rows()).map_err(|_| {
            DodamError::UnsupportedSql("row group row count out of range".to_string())
        })?;
        let key_reader = row_group.get_column_page_reader(key_column)?;
        let sum_reader = row_group.get_column_page_reader(sum_column)?;
        let (mut required_key_cursor, mut nullable_key_cursor) = if nullable_i32_key {
            (
                None,
                Some(NullablePlainPrimitivePageCursor::new(
                    key_reader,
                    key_spec.column_type,
                    key_column_desc.max_def_level(),
                )),
            )
        } else {
            (
                Some(RequiredPlainPrimitivePageCursor::new(
                    key_reader,
                    key_spec.column_type,
                    if optional_all_present_key {
                        key_column_desc.max_def_level()
                    } else {
                        0
                    },
                )),
                None,
            )
        };
        let mut sum_cursor =
            RequiredPlainPrimitivePageCursor::new(sum_reader, sum_spec.column_type, 0);
        let mut rows_read = 0usize;
        while rows_read < row_count {
            let read_started = Instant::now();
            let key_started = Instant::now();
            let key_loaded = if let Some(cursor) = required_key_cursor.as_mut() {
                cursor.ensure_page()?
            } else if let Some(cursor) = nullable_key_cursor.as_mut() {
                cursor.ensure_page()?
            } else {
                false
            };
            if !key_loaded {
                metrics.add_column_read_nanos(0, elapsed_nanos(key_started));
                metrics.add_read_nanos(elapsed_nanos(read_started));
                return Ok(None);
            }
            metrics.add_column_read_nanos(0, elapsed_nanos(key_started));
            let sum_started = Instant::now();
            if !sum_cursor.ensure_page()? {
                metrics.add_column_read_nanos(1, elapsed_nanos(sum_started));
                metrics.add_read_nanos(elapsed_nanos(read_started));
                return Ok(None);
            }
            metrics.add_column_read_nanos(1, elapsed_nanos(sum_started));
            let key_available = required_key_cursor
                .as_ref()
                .map(RequiredPlainPrimitivePageCursor::available_rows)
                .or_else(|| {
                    nullable_key_cursor
                        .as_ref()
                        .map(NullablePlainPrimitivePageCursor::available_rows)
                })
                .unwrap_or(0);
            let rows = batch_size
                .min(row_count - rows_read)
                .min(key_available)
                .min(sum_cursor.available_rows());
            if rows == 0 {
                metrics.add_read_nanos(elapsed_nanos(read_started));
                return Ok(None);
            }
            let Some(sum_bytes) = sum_cursor.raw_bytes(rows) else {
                metrics.add_read_nanos(elapsed_nanos(read_started));
                return Ok(None);
            };
            metrics.add_read_nanos(elapsed_nanos(read_started));
            let consume_started = Instant::now();
            if let Some(cursor) = nullable_key_cursor.as_mut() {
                let Some((key_bytes, key_def_levels)) = cursor.raw_nullable_bytes(rows) else {
                    metrics.add_read_nanos(elapsed_nanos(read_started));
                    return Ok(None);
                };
                consume(DirectPrimitiveCountSumPageBatch::I32NullableI64 {
                    keys: key_bytes,
                    key_def_levels,
                    sums: sum_bytes,
                    rows,
                })?;
            } else {
                let Some(key_bytes) = required_key_cursor
                    .as_ref()
                    .and_then(|cursor| cursor.raw_bytes(rows))
                else {
                    metrics.add_read_nanos(elapsed_nanos(read_started));
                    return Ok(None);
                };
                match key_spec.column_type {
                    DirectPrimitiveColumnType::I32 => {
                        consume(DirectPrimitiveCountSumPageBatch::I32I64 {
                            keys: key_bytes,
                            sums: sum_bytes,
                            rows,
                        })?
                    }
                    DirectPrimitiveColumnType::I64 => {
                        consume(DirectPrimitiveCountSumPageBatch::I64I64 {
                            keys: key_bytes,
                            sums: sum_bytes,
                            rows,
                        })?
                    }
                    _ => return Ok(None),
                }
            }
            metrics.add_consume_nanos(elapsed_nanos(consume_started));
            if let Some(cursor) = required_key_cursor.as_mut() {
                cursor.advance(rows);
            }
            if let Some(cursor) = nullable_key_cursor.as_mut() {
                cursor.advance(rows);
            }
            sum_cursor.advance(rows);
            rows_read += rows;
            metrics.batches += 1;
            metrics.rows = metrics.rows.saturating_add(rows);
        }
    }
    Ok(Some(metrics))
}

fn nullable_count_sum_page_cursor_enabled() -> bool {
    std::env::var("DODAM_ENABLE_NULLABLE_COUNT_SUM_PAGE_CURSOR")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn optional_all_present_count_sum_page_cursor_enabled() -> bool {
    std::env::var("DODAM_ENABLE_OPTIONAL_ALL_PRESENT_COUNT_SUM_PAGE_CURSOR")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

struct RequiredPlainPrimitivePageCursor {
    page_reader: Box<dyn PageReader>,
    column_type: DirectPrimitiveColumnType,
    max_def_level: i16,
    values: Bytes,
    value_offset: usize,
    rows_remaining: usize,
}

impl RequiredPlainPrimitivePageCursor {
    fn new(
        page_reader: Box<dyn PageReader>,
        column_type: DirectPrimitiveColumnType,
        max_def_level: i16,
    ) -> Self {
        Self {
            page_reader,
            column_type,
            max_def_level,
            values: Bytes::new(),
            value_offset: 0,
            rows_remaining: 0,
        }
    }

    fn ensure_page(&mut self) -> Result<bool> {
        if self.rows_remaining > 0 {
            return Ok(true);
        }
        self.load_next_page()
    }

    fn available_rows(&self) -> usize {
        self.rows_remaining
    }

    fn raw_view(&self, records: usize) -> Result<RawColumnView<'_>> {
        if records > self.rows_remaining {
            return Err(DodamError::UnsupportedSql(
                "direct primitive page cursor length mismatch".to_string(),
            ));
        }
        let byte_width = direct_primitive_byte_width(self.column_type);
        let start = self.value_offset.saturating_mul(byte_width);
        let end = start.saturating_add(records.saturating_mul(byte_width));
        if end > self.values.len() {
            return Err(DodamError::UnsupportedSql(
                "direct primitive page cursor byte length mismatch".to_string(),
            ));
        }
        let data = &self.values[start..end];
        Ok(match self.column_type {
            DirectPrimitiveColumnType::I32 => RawColumnView::I32Bytes { data, len: records },
            DirectPrimitiveColumnType::Date32 => RawColumnView::Date32Bytes { data, len: records },
            DirectPrimitiveColumnType::I64 => RawColumnView::I64Bytes { data, len: records },
            DirectPrimitiveColumnType::Decimal128Int64Raw { precision, scale }
            | DirectPrimitiveColumnType::Decimal128Int64 { precision, scale } => {
                RawColumnView::Decimal128I64Bytes {
                    data,
                    len: records,
                    precision,
                    scale,
                }
            }
        })
    }

    fn raw_bytes(&self, records: usize) -> Option<&[u8]> {
        if records > self.rows_remaining {
            return None;
        }
        let byte_width = direct_primitive_byte_width(self.column_type);
        let start = self.value_offset.checked_mul(byte_width)?;
        let end = start.checked_add(records.checked_mul(byte_width)?)?;
        (end <= self.values.len()).then(|| &self.values[start..end])
    }

    fn advance(&mut self, records: usize) {
        self.value_offset += records;
        self.rows_remaining -= records;
    }

    fn load_next_page(&mut self) -> Result<bool> {
        while let Some(page) = self.page_reader.get_next_page()? {
            match page {
                Page::DictionaryPage { .. } => return Ok(false),
                Page::DataPage {
                    buf,
                    num_values,
                    encoding,
                    def_level_encoding,
                    ..
                } => {
                    if encoding != Encoding::PLAIN {
                        return Ok(false);
                    }
                    let mut offset = 0usize;
                    if self.max_def_level > 0 {
                        if def_level_encoding != Encoding::RLE {
                            return Ok(false);
                        }
                        let (bytes_read, level_data) =
                            parse_v1_rle_level_data(buf.slice(offset..))?;
                        offset += bytes_read;
                        if !plain_page_all_present(
                            level_data,
                            self.max_def_level,
                            num_values as usize,
                        )? {
                            return Ok(false);
                        }
                    }
                    if offset > buf.len() {
                        return Ok(false);
                    }
                    self.values = buf.slice(offset..);
                    self.value_offset = 0;
                    self.rows_remaining = num_values as usize;
                    return Ok(true);
                }
                Page::DataPageV2 {
                    buf,
                    num_values,
                    encoding,
                    num_nulls,
                    rep_levels_byte_len,
                    def_levels_byte_len,
                    ..
                } => {
                    if encoding != Encoding::PLAIN {
                        return Ok(false);
                    }
                    if self.max_def_level > 0 && num_nulls > 0 {
                        let def_start = rep_levels_byte_len as usize;
                        let def_end = def_start + def_levels_byte_len as usize;
                        if def_end > buf.len()
                            || !plain_page_all_present(
                                buf.slice(def_start..def_end),
                                self.max_def_level,
                                num_values as usize,
                            )?
                        {
                            return Ok(false);
                        }
                    }
                    let value_start = (rep_levels_byte_len + def_levels_byte_len) as usize;
                    if value_start > buf.len() {
                        return Ok(false);
                    }
                    self.values = buf.slice(value_start..);
                    self.value_offset = 0;
                    self.rows_remaining = num_values as usize;
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }
}

struct NullablePlainPrimitivePageCursor {
    page_reader: Box<dyn PageReader>,
    column_type: DirectPrimitiveColumnType,
    max_def_level: i16,
    values: Bytes,
    def_levels: Vec<i16>,
    value_offset: usize,
    row_offset: usize,
    rows_remaining: usize,
}

impl NullablePlainPrimitivePageCursor {
    fn new(
        page_reader: Box<dyn PageReader>,
        column_type: DirectPrimitiveColumnType,
        max_def_level: i16,
    ) -> Self {
        Self {
            page_reader,
            column_type,
            max_def_level,
            values: Bytes::new(),
            def_levels: Vec::new(),
            value_offset: 0,
            row_offset: 0,
            rows_remaining: 0,
        }
    }

    fn ensure_page(&mut self) -> Result<bool> {
        if self.rows_remaining > 0 {
            return Ok(true);
        }
        self.load_next_page()
    }

    fn available_rows(&self) -> usize {
        self.rows_remaining
    }

    fn raw_nullable_bytes(&self, records: usize) -> Option<(&[u8], &[i16])> {
        if records > self.rows_remaining {
            return None;
        }
        let def_end = self.row_offset.checked_add(records)?;
        let def_slice = self.def_levels.get(self.row_offset..def_end)?;
        let present = count_present_def_levels(def_slice, self.max_def_level);
        let byte_width = direct_primitive_byte_width(self.column_type);
        let start = self.value_offset.checked_mul(byte_width)?;
        let end = start.checked_add(present.checked_mul(byte_width)?)?;
        (end <= self.values.len()).then(|| (&self.values[start..end], def_slice))
    }

    fn advance(&mut self, records: usize) {
        let def_end = self.row_offset + records;
        let present = count_present_def_levels(
            &self.def_levels[self.row_offset..def_end],
            self.max_def_level,
        );
        self.value_offset += present;
        self.row_offset = def_end;
        self.rows_remaining -= records;
    }

    fn load_next_page(&mut self) -> Result<bool> {
        while let Some(page) = self.page_reader.get_next_page()? {
            match page {
                Page::DictionaryPage { .. } => return Ok(false),
                Page::DataPage {
                    buf,
                    num_values,
                    encoding,
                    def_level_encoding,
                    ..
                } => {
                    if encoding != Encoding::PLAIN
                        || def_level_encoding != Encoding::RLE
                        || self.max_def_level <= 0
                    {
                        return Ok(false);
                    }
                    let mut offset = 0usize;
                    let (bytes_read, level_data) = parse_v1_rle_level_data(buf.slice(offset..))?;
                    offset += bytes_read;
                    self.def_levels.clear();
                    decode_rle_i16_values(
                        level_data,
                        num_required_bits_i16(self.max_def_level),
                        num_values as usize,
                        &mut self.def_levels,
                    )?;
                    if self.def_levels.len() != num_values as usize || offset > buf.len() {
                        return Ok(false);
                    }
                    self.values = buf.slice(offset..);
                    self.value_offset = 0;
                    self.row_offset = 0;
                    self.rows_remaining = num_values as usize;
                    return Ok(true);
                }
                Page::DataPageV2 {
                    buf,
                    num_values,
                    encoding,
                    num_nulls,
                    rep_levels_byte_len,
                    def_levels_byte_len,
                    ..
                } => {
                    if encoding != Encoding::PLAIN || self.max_def_level <= 0 {
                        return Ok(false);
                    }
                    let def_start = rep_levels_byte_len as usize;
                    let def_end = def_start + def_levels_byte_len as usize;
                    if def_end > buf.len() {
                        return Ok(false);
                    }
                    self.def_levels.clear();
                    if num_nulls == 0 {
                        self.def_levels
                            .resize(num_values as usize, self.max_def_level);
                    } else {
                        decode_rle_i16_values(
                            buf.slice(def_start..def_end),
                            num_required_bits_i16(self.max_def_level),
                            num_values as usize,
                            &mut self.def_levels,
                        )?;
                    }
                    if self.def_levels.len() != num_values as usize {
                        return Ok(false);
                    }
                    let value_start = (rep_levels_byte_len + def_levels_byte_len) as usize;
                    if value_start > buf.len() {
                        return Ok(false);
                    }
                    self.values = buf.slice(value_start..);
                    self.value_offset = 0;
                    self.row_offset = 0;
                    self.rows_remaining = num_values as usize;
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }
}

fn direct_primitive_byte_width(column_type: DirectPrimitiveColumnType) -> usize {
    match column_type {
        DirectPrimitiveColumnType::I32 | DirectPrimitiveColumnType::Date32 => 4,
        DirectPrimitiveColumnType::I64
        | DirectPrimitiveColumnType::Decimal128Int64 { .. }
        | DirectPrimitiveColumnType::Decimal128Int64Raw { .. } => 8,
    }
}

fn scan_parquet_primitive_columns_page_reader<R, F>(
    reader: SerializedFileReader<R>,
    batch_size: usize,
    row_groups: &[usize],
    columns: &[DirectPrimitiveColumnSpec<'_>],
    consume: &mut F,
) -> Result<Option<DirectPrimitiveColumnScanMetrics>>
where
    R: ChunkReader + 'static,
    F: for<'a> FnMut(&[RawColumnView<'a>]) -> Result<()>,
{
    let names: Vec<&str> = columns.iter().map(|column| column.name).collect();
    let Some(column_indices) = parquet_column_indices_by_name(&reader, &names) else {
        return Ok(None);
    };
    let schema = reader.metadata().file_metadata().schema_descr();
    let required_columns: Vec<bool> = column_indices
        .iter()
        .map(|&index| schema.column(index).max_def_level() == 0)
        .collect();
    let mut metrics = DirectPrimitiveColumnScanMetrics {
        row_groups: row_groups.len(),
        column_read_nanos: vec![0; columns.len()],
        ..DirectPrimitiveColumnScanMetrics::default()
    };
    for &row_group_index in row_groups {
        let row_group = reader.get_row_group(row_group_index)?;
        let row_count = usize::try_from(row_group.metadata().num_rows()).map_err(|_| {
            DodamError::UnsupportedSql("row group row count out of range".to_string())
        })?;
        let mut values = Vec::with_capacity(columns.len());
        let mut def_levels = Vec::with_capacity(columns.len());
        let read_started = Instant::now();
        for (index, (column, &column_index)) in
            columns.iter().zip(column_indices.iter()).enumerate()
        {
            let column_started = Instant::now();
            let Some((column_values, column_def_levels, column_rows)) =
                read_direct_primitive_page_column(
                    &*row_group,
                    column_index,
                    column.column_type,
                    row_count,
                )?
            else {
                metrics.add_read_nanos(elapsed_nanos(read_started));
                return Ok(None);
            };
            metrics.add_column_read_nanos(index, elapsed_nanos(column_started));
            if column_rows != row_count {
                metrics.add_read_nanos(elapsed_nanos(read_started));
                return Ok(None);
            }
            values.push(column_values);
            def_levels.push(column_def_levels);
        }
        metrics.add_read_nanos(elapsed_nanos(read_started));
        let mut value_offsets = vec![0usize; columns.len()];
        let mut row_offset = 0usize;
        while row_offset < row_count {
            let records = batch_size.min(row_count - row_offset);
            let consume_started = Instant::now();
            let views = direct_primitive_page_batch_views(
                &values,
                columns,
                &required_columns,
                &def_levels,
                &mut value_offsets,
                row_offset,
                records,
            )?;
            consume(&views)?;
            metrics.add_consume_nanos(elapsed_nanos(consume_started));
            metrics.batches += 1;
            metrics.rows = metrics.rows.saturating_add(records);
            row_offset += records;
        }
    }
    Ok(Some(metrics))
}

fn direct_primitive_page_batch_views<'a>(
    values: &'a [DirectPrimitiveColumnValues],
    columns: &[DirectPrimitiveColumnSpec<'_>],
    required_columns: &[bool],
    def_levels: &'a [Vec<i16>],
    value_offsets: &mut [usize],
    row_offset: usize,
    records: usize,
) -> Result<Vec<RawColumnView<'a>>> {
    values
        .iter()
        .zip(columns.iter())
        .zip(required_columns.iter())
        .zip(def_levels.iter())
        .zip(value_offsets.iter_mut())
        .map(
            |((((values, column), required), def_levels), value_offset)| {
                direct_primitive_page_batch_view(
                    values,
                    column.column_type,
                    *required,
                    def_levels,
                    value_offset,
                    row_offset,
                    records,
                )
            },
        )
        .collect()
}

fn direct_primitive_page_batch_view<'a>(
    values: &'a DirectPrimitiveColumnValues,
    column_type: DirectPrimitiveColumnType,
    required: bool,
    def_levels: &'a [i16],
    value_offset: &mut usize,
    row_offset: usize,
    records: usize,
) -> Result<RawColumnView<'a>> {
    if required || def_levels.is_empty() {
        let start = row_offset;
        let end = row_offset + records;
        *value_offset = end;
        return direct_primitive_non_null_page_batch_view(values, column_type, start, end);
    }
    let def_end = row_offset + records;
    if def_end > def_levels.len() {
        return Err(DodamError::UnsupportedSql(
            "direct primitive page def-level length mismatch".to_string(),
        ));
    }
    let present = count_present_def_levels(&def_levels[row_offset..def_end], 1);
    let start = *value_offset;
    let end = start + present;
    *value_offset = end;
    match (values, column_type) {
        (DirectPrimitiveColumnValues::I64(values), DirectPrimitiveColumnType::I64) => {
            if end > values.len() {
                return Err(DodamError::UnsupportedSql(
                    "direct primitive page value length mismatch".to_string(),
                ));
            }
            Ok(RawColumnView::I64Nullable {
                values: &values[start..end],
                def_levels: &def_levels[row_offset..def_end],
            })
        }
        (DirectPrimitiveColumnValues::I32(values), DirectPrimitiveColumnType::I32) => {
            if end > values.len() {
                return Err(DodamError::UnsupportedSql(
                    "direct primitive page value length mismatch".to_string(),
                ));
            }
            Ok(RawColumnView::I32Nullable {
                values: &values[start..end],
                def_levels: &def_levels[row_offset..def_end],
            })
        }
        _ => direct_primitive_non_null_page_batch_view(values, column_type, start, end),
    }
}

fn direct_primitive_non_null_page_batch_view<'a>(
    values: &'a DirectPrimitiveColumnValues,
    column_type: DirectPrimitiveColumnType,
    start: usize,
    end: usize,
) -> Result<RawColumnView<'a>> {
    match (values, column_type) {
        (DirectPrimitiveColumnValues::I64(values), DirectPrimitiveColumnType::I64) => {
            if end > values.len() {
                return Err(DodamError::UnsupportedSql(
                    "direct primitive page value length mismatch".to_string(),
                ));
            }
            Ok(RawColumnView::I64(&values[start..end]))
        }
        (DirectPrimitiveColumnValues::I32(values), DirectPrimitiveColumnType::I32) => {
            if end > values.len() {
                return Err(DodamError::UnsupportedSql(
                    "direct primitive page value length mismatch".to_string(),
                ));
            }
            Ok(RawColumnView::I32(&values[start..end]))
        }
        (DirectPrimitiveColumnValues::I32(values), DirectPrimitiveColumnType::Date32) => {
            if end > values.len() {
                return Err(DodamError::UnsupportedSql(
                    "direct primitive page value length mismatch".to_string(),
                ));
            }
            Ok(RawColumnView::Date32(&values[start..end]))
        }
        (
            DirectPrimitiveColumnValues::Decimal128 { values, .. },
            DirectPrimitiveColumnType::Decimal128Int64 { precision, scale },
        ) => {
            if end > values.len() {
                return Err(DodamError::UnsupportedSql(
                    "direct primitive page value length mismatch".to_string(),
                ));
            }
            Ok(RawColumnView::Decimal128 {
                values: &values[start..end],
                precision,
                scale,
            })
        }
        (
            DirectPrimitiveColumnValues::I64(values),
            DirectPrimitiveColumnType::Decimal128Int64Raw { precision, scale },
        ) => {
            if end > values.len() {
                return Err(DodamError::UnsupportedSql(
                    "direct primitive page value length mismatch".to_string(),
                ));
            }
            Ok(RawColumnView::Decimal128I64 {
                values: &values[start..end],
                precision,
                scale,
            })
        }
        _ => Err(DodamError::UnsupportedSql(
            "direct primitive page batch view type mismatch".to_string(),
        )),
    }
}

fn read_direct_primitive_page_column(
    row_group: &dyn parquet::file::reader::RowGroupReader,
    column_index: usize,
    column_type: DirectPrimitiveColumnType,
    row_count: usize,
) -> Result<Option<(DirectPrimitiveColumnValues, Vec<i16>, usize)>> {
    let column_desc = row_group.metadata().schema_descr().column(column_index);
    if column_desc.max_rep_level() != 0 || column_desc.max_def_level() > 1 {
        return Ok(None);
    }
    if column_desc.max_def_level() > 0
        && !matches!(
            column_type,
            DirectPrimitiveColumnType::I32 | DirectPrimitiveColumnType::I64
        )
    {
        return Ok(None);
    }
    if !direct_primitive_page_column_type_matches(column_desc.physical_type(), column_type) {
        return Ok(None);
    }
    let mut values = DirectPrimitiveColumnValues::new(column_type, row_count);
    let mut def_levels =
        Vec::<i16>::with_capacity((column_desc.max_def_level() > 0) as usize * row_count);
    let mut present_total = 0usize;
    let mut rows = 0usize;
    let mut page_reader = row_group.get_column_page_reader(column_index)?;
    while let Some(page) = page_reader.get_next_page()? {
        match page {
            Page::DictionaryPage { .. } => return Ok(None),
            Page::DataPage {
                buf,
                num_values,
                encoding,
                def_level_encoding,
                ..
            } => {
                if encoding != Encoding::PLAIN {
                    return Ok(None);
                }
                let page_rows = num_values as usize;
                let mut offset = 0usize;
                let present_values = if column_desc.max_def_level() > 0 {
                    if def_level_encoding != Encoding::RLE {
                        return Ok(None);
                    }
                    let (bytes_read, level_data) = parse_v1_rle_level_data(buf.slice(offset..))?;
                    offset += bytes_read;
                    let start = def_levels.len();
                    decode_rle_i16_values(
                        level_data,
                        num_required_bits_i16(column_desc.max_def_level()),
                        page_rows,
                        &mut def_levels,
                    )?;
                    def_levels[start..]
                        .iter()
                        .filter(|level| **level == column_desc.max_def_level())
                        .count()
                } else {
                    page_rows
                };
                decode_plain_direct_primitive_values(
                    buf.slice(offset..),
                    present_values,
                    column_type,
                    &mut values,
                )?;
                present_total = present_total.saturating_add(present_values);
                rows = rows.saturating_add(page_rows);
            }
            Page::DataPageV2 {
                buf,
                num_values,
                encoding,
                num_nulls,
                def_levels_byte_len,
                rep_levels_byte_len,
                ..
            } => {
                if encoding != Encoding::PLAIN {
                    return Ok(None);
                }
                let page_rows = num_values as usize;
                let present_values = if column_desc.max_def_level() > 0 {
                    let def_start = rep_levels_byte_len as usize;
                    let def_end = def_start + def_levels_byte_len as usize;
                    if def_end > buf.len() {
                        return Ok(None);
                    }
                    if num_nulls == 0 {
                        def_levels
                            .resize(def_levels.len() + page_rows, column_desc.max_def_level());
                        page_rows
                    } else {
                        let start = def_levels.len();
                        decode_rle_i16_values(
                            buf.slice(def_start..def_end),
                            num_required_bits_i16(column_desc.max_def_level()),
                            page_rows,
                            &mut def_levels,
                        )?;
                        let present_values = def_levels[start..]
                            .iter()
                            .filter(|level| **level == column_desc.max_def_level())
                            .count();
                        if present_values != (num_values - num_nulls) as usize {
                            return Ok(None);
                        }
                        present_values
                    }
                } else {
                    page_rows
                };
                let value_start = (rep_levels_byte_len + def_levels_byte_len) as usize;
                if value_start > buf.len() {
                    return Ok(None);
                }
                decode_plain_direct_primitive_values(
                    buf.slice(value_start..),
                    present_values,
                    column_type,
                    &mut values,
                )?;
                present_total = present_total.saturating_add(present_values);
                rows = rows.saturating_add(page_rows);
            }
        }
    }
    if rows != row_count {
        return Ok(None);
    }
    if column_desc.max_def_level() > 0 && def_levels.len() != row_count {
        return Ok(None);
    }
    if values.value_count() != present_total {
        return Ok(None);
    }
    Ok(Some((values, def_levels, rows)))
}

fn direct_primitive_page_column_type_matches(
    physical_type: ParquetPhysicalType,
    column_type: DirectPrimitiveColumnType,
) -> bool {
    matches!(
        (physical_type, column_type),
        (ParquetPhysicalType::INT32, DirectPrimitiveColumnType::I32)
            | (
                ParquetPhysicalType::INT32,
                DirectPrimitiveColumnType::Date32
            )
            | (ParquetPhysicalType::INT64, DirectPrimitiveColumnType::I64)
            | (
                ParquetPhysicalType::INT64,
                DirectPrimitiveColumnType::Decimal128Int64 { .. }
                    | DirectPrimitiveColumnType::Decimal128Int64Raw { .. }
            )
    )
}

fn decode_plain_direct_primitive_values(
    data: Bytes,
    values: usize,
    column_type: DirectPrimitiveColumnType,
    output: &mut DirectPrimitiveColumnValues,
) -> Result<()> {
    match (column_type, output) {
        (
            DirectPrimitiveColumnType::I32 | DirectPrimitiveColumnType::Date32,
            DirectPrimitiveColumnValues::I32(output),
        ) => decode_plain_i32_values(data, values, output),
        (DirectPrimitiveColumnType::I64, DirectPrimitiveColumnValues::I64(output)) => {
            decode_plain_i64_values(data, values, output)
        }
        (
            DirectPrimitiveColumnType::Decimal128Int64 { .. },
            DirectPrimitiveColumnValues::Decimal128 {
                values: output,
                raw_i64,
            },
        ) => {
            let before = raw_i64.len();
            decode_plain_i64_values(data, values, raw_i64)?;
            output.extend(raw_i64[before..].iter().copied().map(i128::from));
            Ok(())
        }
        (
            DirectPrimitiveColumnType::Decimal128Int64Raw { .. },
            DirectPrimitiveColumnValues::I64(output),
        ) => decode_plain_i64_values(data, values, output),
        _ => Err(DodamError::UnsupportedSql(
            "direct primitive page decoder type mismatch".to_string(),
        )),
    }
}

fn decode_plain_i32_values(data: Bytes, values: usize, output: &mut Vec<i32>) -> Result<()> {
    let byte_len = values.saturating_mul(std::mem::size_of::<i32>());
    if data.len() < byte_len {
        return Ok(());
    }
    #[cfg(target_endian = "little")]
    {
        let before = output.len();
        output.reserve(values);
        // Parquet PLAIN int32 is little-endian fixed width. On little-endian hosts this is a byte copy.
        unsafe {
            output.set_len(before + values);
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                output.as_mut_ptr().add(before).cast::<u8>(),
                byte_len,
            );
        }
        return Ok(());
    }
    #[cfg(not(target_endian = "little"))]
    {
        output.reserve(values);
        for chunk in data[..byte_len].chunks_exact(4) {
            output.push(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        Ok(())
    }
}

fn decode_plain_i64_values(data: Bytes, values: usize, output: &mut Vec<i64>) -> Result<()> {
    let byte_len = values.saturating_mul(std::mem::size_of::<i64>());
    if data.len() < byte_len {
        return Ok(());
    }
    #[cfg(target_endian = "little")]
    {
        let before = output.len();
        output.reserve(values);
        // Parquet PLAIN int64 is little-endian fixed width. On little-endian hosts this is a byte copy.
        unsafe {
            output.set_len(before + values);
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                output.as_mut_ptr().add(before).cast::<u8>(),
                byte_len,
            );
        }
        return Ok(());
    }
    #[cfg(not(target_endian = "little"))]
    {
        output.reserve(values);
        for chunk in data[..byte_len].chunks_exact(8) {
            output.push(i64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ]));
        }
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn build_selected_runs(
    decimals: &[i64],
    dates: &[i32],
    decimal_min: Option<i64>,
    decimal_max: Option<i64>,
    date_min: Option<i32>,
    date_max: Option<i32>,
    runs: &mut Vec<(usize, usize)>,
    selected_decimals: &mut Vec<i64>,
    selected_dates: &mut Vec<i32>,
) {
    let mut run_start = None;
    let mut run_len = 0usize;
    for row in 0..decimals.len() {
        let selected = decimal_min.is_none_or(|min| decimals[row] >= min)
            && decimal_max.is_none_or(|max| decimals[row] <= max)
            && date_min.is_none_or(|min| dates[row] >= min)
            && date_max.is_none_or(|max| dates[row] <= max);
        if selected {
            selected_decimals.push(decimals[row]);
            selected_dates.push(dates[row]);
            if run_start.is_none() {
                run_start = Some(row);
                run_len = 1;
            } else {
                run_len += 1;
            }
        } else if let Some(start) = run_start.take() {
            runs.push((start, run_len));
            run_len = 0;
        }
    }
    if let Some(start) = run_start {
        runs.push((start, run_len));
    }
}

fn read_i32_selected_runs(
    reader: &mut ColumnReaderImpl<Int32Type>,
    records: usize,
    runs: &[(usize, usize)],
    required: bool,
    def_levels: &mut Vec<i16>,
    output: &mut Vec<i32>,
    metrics: &mut DirectPrimitiveColumnScanMetrics,
) -> Result<bool> {
    let mut cursor = 0usize;
    for &(start, len) in runs {
        if start > cursor {
            let skipped = start - cursor;
            metrics.add_selected_skip(skipped);
            if reader.skip_records(skipped)? != skipped {
                return Ok(false);
            }
        }
        metrics.add_selected_read(len);
        if len == 0 {
            cursor = start;
            continue;
        }
        def_levels.clear();
        let (read_records, value_count, level_count) =
            reader.read_records(len, (!required).then_some(&mut *def_levels), None, output)?;
        if read_records != len
            || value_count != len
            || !direct_def_levels_match(level_count, len, required)
            || !direct_all_present(required, def_levels)
        {
            return Ok(false);
        }
        cursor = start + len;
    }
    if records > cursor {
        let skipped = records - cursor;
        metrics.add_selected_skip(skipped);
        if reader.skip_records(skipped)? != skipped {
            return Ok(false);
        }
    }
    Ok(true)
}

fn read_i32_plain_page_selected_runs(
    row_group: &dyn parquet::file::reader::RowGroupReader,
    column: usize,
    records: usize,
    runs: &[(usize, usize)],
    output: &mut Vec<i32>,
    metrics: &mut DirectPrimitiveColumnScanMetrics,
) -> Result<Option<()>> {
    let column_desc = row_group.metadata().schema_descr().column(column);
    if column_desc.physical_type() != ParquetPhysicalType::INT32
        || column_desc.max_rep_level() != 0
        || column_desc.max_def_level() != 0
    {
        return Ok(None);
    }
    let mut page_reader = row_group.get_column_page_reader(column)?;
    let mut page_row_start = 0usize;
    while let Some(page) = page_reader.get_next_page()? {
        match page {
            Page::DictionaryPage { .. } => return Ok(None),
            Page::DataPage {
                buf,
                num_values,
                encoding,
                ..
            } => {
                if encoding != Encoding::PLAIN {
                    return Ok(None);
                }
                let page_rows = num_values as usize;
                compact_plain_i32_page_selected_ranges(
                    buf,
                    0,
                    page_row_start,
                    page_rows,
                    runs,
                    output,
                    metrics,
                )?;
                page_row_start += page_rows;
            }
            Page::DataPageV2 {
                buf,
                num_values,
                encoding,
                def_levels_byte_len,
                rep_levels_byte_len,
                ..
            } => {
                if encoding != Encoding::PLAIN {
                    return Ok(None);
                }
                let page_rows = num_values as usize;
                let value_start = (rep_levels_byte_len + def_levels_byte_len) as usize;
                compact_plain_i32_page_selected_ranges(
                    buf,
                    value_start,
                    page_row_start,
                    page_rows,
                    runs,
                    output,
                    metrics,
                )?;
                page_row_start += page_rows;
            }
        }
    }
    if page_row_start != records {
        return Ok(None);
    }
    Ok(Some(()))
}

fn read_i64_plain_page_selected_runs(
    row_group: &dyn parquet::file::reader::RowGroupReader,
    column: usize,
    records: usize,
    runs: &[(usize, usize)],
    output: &mut Vec<i64>,
    metrics: &mut DirectPrimitiveColumnScanMetrics,
) -> Result<Option<()>> {
    let column_desc = row_group.metadata().schema_descr().column(column);
    if column_desc.physical_type() != ParquetPhysicalType::INT64
        || column_desc.max_rep_level() != 0
        || column_desc.max_def_level() > 1
    {
        return Ok(None);
    }
    let mut page_reader = row_group.get_column_page_reader(column)?;
    let mut page_row_start = 0usize;
    let mut run_cursor = 0usize;
    while let Some(page) = page_reader.get_next_page()? {
        match page {
            Page::DictionaryPage { .. } => return Ok(None),
            Page::DataPage {
                buf,
                num_values,
                encoding,
                def_level_encoding,
                ..
            } => {
                if encoding != Encoding::PLAIN {
                    return Ok(None);
                }
                let page_rows = num_values as usize;
                let page_row_end = page_row_start + page_rows;
                advance_run_cursor(runs, &mut run_cursor, page_row_start);
                if !runs_overlap_from(runs, run_cursor, page_row_start, page_row_end) {
                    page_row_start = page_row_end;
                    continue;
                }
                let mut value_start = 0usize;
                if column_desc.max_def_level() > 0 {
                    let (bytes_read, level_data) = parse_v1_rle_level_data(buf.slice(0..))?;
                    value_start = bytes_read;
                    if def_level_encoding != Encoding::RLE {
                        return Ok(None);
                    }
                    if !plain_page_all_present(level_data, column_desc.max_def_level(), page_rows)?
                    {
                        return Ok(None);
                    }
                }
                compact_plain_i64_page_selected_ranges(
                    buf,
                    value_start,
                    page_row_start,
                    page_rows,
                    runs,
                    output,
                    metrics,
                )?;
                page_row_start += page_rows;
            }
            Page::DataPageV2 {
                buf,
                num_values,
                encoding,
                num_nulls,
                def_levels_byte_len,
                rep_levels_byte_len,
                ..
            } => {
                if encoding != Encoding::PLAIN {
                    return Ok(None);
                }
                let page_rows = num_values as usize;
                let page_row_end = page_row_start + page_rows;
                advance_run_cursor(runs, &mut run_cursor, page_row_start);
                if !runs_overlap_from(runs, run_cursor, page_row_start, page_row_end) {
                    page_row_start = page_row_end;
                    continue;
                }
                if column_desc.max_def_level() > 0 {
                    if num_nulls != 0 {
                        return Ok(None);
                    }
                    let def_start = rep_levels_byte_len as usize;
                    let def_end = def_start + def_levels_byte_len as usize;
                    if def_end > buf.len() {
                        return Ok(None);
                    }
                }
                let value_start = (rep_levels_byte_len + def_levels_byte_len) as usize;
                compact_plain_i64_page_selected_ranges(
                    buf,
                    value_start,
                    page_row_start,
                    page_rows,
                    runs,
                    output,
                    metrics,
                )?;
                page_row_start += page_rows;
            }
        }
    }
    if page_row_start != records {
        return Ok(None);
    }
    Ok(Some(()))
}

fn compact_plain_i32_page_selected_ranges(
    buf: Bytes,
    value_start: usize,
    page_row_start: usize,
    page_rows: usize,
    runs: &[(usize, usize)],
    output: &mut Vec<i32>,
    metrics: &mut DirectPrimitiveColumnScanMetrics,
) -> Result<()> {
    let page_row_end = page_row_start + page_rows;
    if !runs_overlap(runs, page_row_start, page_row_end) {
        return Ok(());
    }
    let value_bytes = page_rows.saturating_mul(std::mem::size_of::<i32>());
    if value_start.saturating_add(value_bytes) > buf.len() {
        return Err(DodamError::UnsupportedSql(
            "selected i32 page payload length mismatch".to_string(),
        ));
    }
    for &(run_start, run_len) in runs {
        let run_end = run_start + run_len;
        if run_start >= page_row_end {
            break;
        }
        if run_end <= page_row_start {
            continue;
        }
        let local_start = run_start.max(page_row_start) - page_row_start;
        let local_end = run_end.min(page_row_end) - page_row_start;
        let rows = local_end - local_start;
        let byte_start = value_start + local_start * std::mem::size_of::<i32>();
        let byte_end = value_start + local_end * std::mem::size_of::<i32>();
        metrics.add_selected_read(rows);
        decode_plain_i32_values(buf.slice(byte_start..byte_end), rows, output)?;
    }
    Ok(())
}

fn plain_page_all_present(data: Bytes, max_def_level: i16, rows: usize) -> Result<bool> {
    rle_i16_all_equal(
        data,
        num_required_bits_i16(max_def_level),
        rows,
        max_def_level,
    )
}

fn compact_plain_i64_page_selected_ranges(
    buf: Bytes,
    value_start: usize,
    page_row_start: usize,
    page_rows: usize,
    runs: &[(usize, usize)],
    output: &mut Vec<i64>,
    metrics: &mut DirectPrimitiveColumnScanMetrics,
) -> Result<()> {
    let page_row_end = page_row_start + page_rows;
    if !runs_overlap(runs, page_row_start, page_row_end) {
        return Ok(());
    }
    let value_bytes = page_rows.saturating_mul(std::mem::size_of::<i64>());
    if value_start.saturating_add(value_bytes) > buf.len() {
        return Err(DodamError::UnsupportedSql(
            "selected i64 page payload length mismatch".to_string(),
        ));
    }
    for &(run_start, run_len) in runs {
        let run_end = run_start + run_len;
        if run_start >= page_row_end {
            break;
        }
        if run_end <= page_row_start {
            continue;
        }
        let local_start = run_start.max(page_row_start) - page_row_start;
        let local_end = run_end.min(page_row_end) - page_row_start;
        let rows = local_end - local_start;
        let byte_start = value_start + local_start * std::mem::size_of::<i64>();
        metrics.add_selected_read(rows);
        copy_plain_i64_bytes_to_output(&buf, byte_start, rows, output);
    }
    Ok(())
}

fn copy_plain_i64_bytes_to_output(
    buf: &Bytes,
    byte_start: usize,
    rows: usize,
    output: &mut Vec<i64>,
) {
    if rows == 0 {
        return;
    }
    #[cfg(target_endian = "little")]
    {
        let byte_len = rows.saturating_mul(std::mem::size_of::<i64>());
        let before = output.len();
        output.reserve(rows);
        unsafe {
            output.set_len(before + rows);
            std::ptr::copy_nonoverlapping(
                buf.as_ptr().add(byte_start),
                output.as_mut_ptr().add(before).cast::<u8>(),
                byte_len,
            );
        }
    }
    #[cfg(not(target_endian = "little"))]
    {
        let byte_end = byte_start + rows.saturating_mul(std::mem::size_of::<i64>());
        for chunk in buf[byte_start..byte_end].chunks_exact(8) {
            output.push(i64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ]));
        }
    }
}

fn read_i64_selected_runs(
    reader: &mut ColumnReaderImpl<Int64Type>,
    records: usize,
    runs: &[(usize, usize)],
    required: bool,
    def_levels: &mut Vec<i16>,
    output: &mut Vec<i64>,
    metrics: &mut DirectPrimitiveColumnScanMetrics,
) -> Result<bool> {
    let mut cursor = 0usize;
    for &(start, len) in runs {
        if start > cursor {
            let skipped = start - cursor;
            metrics.add_selected_skip(skipped);
            if reader.skip_records(skipped)? != skipped {
                return Ok(false);
            }
        }
        metrics.add_selected_read(len);
        if len == 0 {
            cursor = start;
            continue;
        }
        def_levels.clear();
        let (read_records, value_count, level_count) =
            reader.read_records(len, (!required).then_some(&mut *def_levels), None, output)?;
        if read_records != len
            || value_count != len
            || !direct_def_levels_match(level_count, len, required)
            || !direct_all_present(required, def_levels)
        {
            return Ok(false);
        }
        cursor = start + len;
    }
    if records > cursor {
        let skipped = records - cursor;
        metrics.add_selected_skip(skipped);
        if reader.skip_records(skipped)? != skipped {
            return Ok(false);
        }
    }
    Ok(true)
}

fn read_i32_selected_windows(
    reader: &mut ColumnReaderImpl<Int32Type>,
    records: usize,
    runs: &[(usize, usize)],
    windows: &[(usize, usize)],
    required: bool,
    def_levels: &mut Vec<i16>,
    scratch: &mut Vec<i32>,
    output: &mut Vec<i32>,
    metrics: &mut DirectPrimitiveColumnScanMetrics,
) -> Result<bool> {
    let mut cursor = 0usize;
    let mut run_index = 0usize;
    for &(window_start, window_len) in windows {
        if window_start > cursor {
            let skipped = window_start - cursor;
            metrics.add_selected_skip(skipped);
            if reader.skip_records(skipped)? != skipped {
                return Ok(false);
            }
        }
        metrics.add_selected_read(window_len);
        scratch.clear();
        def_levels.clear();
        let (read_records, value_count, level_count) = reader.read_records(
            window_len,
            (!required).then_some(&mut *def_levels),
            None,
            scratch,
        )?;
        if read_records != window_len
            || value_count != window_len
            || !direct_def_levels_match(level_count, window_len, required)
            || !direct_all_present(required, def_levels)
        {
            return Ok(false);
        }
        let window_end = window_start + window_len;
        while run_index < runs.len() && runs[run_index].0 < window_end {
            let (run_start, run_len) = runs[run_index];
            let run_end = run_start + run_len;
            if run_end > window_start {
                let start = run_start.max(window_start) - window_start;
                let end = run_end.min(window_end) - window_start;
                output.extend_from_slice(&scratch[start..end]);
            }
            if run_end <= window_end {
                run_index += 1;
            } else {
                break;
            }
        }
        cursor = window_end;
    }
    if records > cursor {
        let skipped = records - cursor;
        metrics.add_selected_skip(skipped);
        if reader.skip_records(skipped)? != skipped {
            return Ok(false);
        }
    }
    Ok(true)
}

fn read_i64_selected_windows(
    reader: &mut ColumnReaderImpl<Int64Type>,
    records: usize,
    runs: &[(usize, usize)],
    windows: &[(usize, usize)],
    required: bool,
    def_levels: &mut Vec<i16>,
    scratch: &mut Vec<i64>,
    output: &mut Vec<i64>,
    metrics: &mut DirectPrimitiveColumnScanMetrics,
) -> Result<bool> {
    let mut cursor = 0usize;
    let mut run_index = 0usize;
    for &(window_start, window_len) in windows {
        if window_start > cursor {
            let skipped = window_start - cursor;
            metrics.add_selected_skip(skipped);
            if reader.skip_records(skipped)? != skipped {
                return Ok(false);
            }
        }
        metrics.add_selected_read(window_len);
        scratch.clear();
        def_levels.clear();
        let (read_records, value_count, level_count) = reader.read_records(
            window_len,
            (!required).then_some(&mut *def_levels),
            None,
            scratch,
        )?;
        if read_records != window_len
            || value_count != window_len
            || !direct_def_levels_match(level_count, window_len, required)
            || !direct_all_present(required, def_levels)
        {
            return Ok(false);
        }
        let window_end = window_start + window_len;
        while run_index < runs.len() && runs[run_index].0 < window_end {
            let (run_start, run_len) = runs[run_index];
            let run_end = run_start + run_len;
            if run_end > window_start {
                let start = run_start.max(window_start) - window_start;
                let end = run_end.min(window_end) - window_start;
                output.extend_from_slice(&scratch[start..end]);
            }
            if run_end <= window_end {
                run_index += 1;
            } else {
                break;
            }
        }
        cursor = window_end;
    }
    if records > cursor {
        let skipped = records - cursor;
        metrics.add_selected_skip(skipped);
        if reader.skip_records(skipped)? != skipped {
            return Ok(false);
        }
    }
    Ok(true)
}

fn compact_selected_i32(values: &[i32], runs: &[(usize, usize)], output: &mut Vec<i32>) {
    for &(start, len) in runs {
        output.extend_from_slice(&values[start..start + len]);
    }
}

fn compact_selected_i64(values: &[i64], runs: &[(usize, usize)], output: &mut Vec<i64>) {
    for &(start, len) in runs {
        output.extend_from_slice(&values[start..start + len]);
    }
}

fn coalesce_selected_runs_to_windows(runs: &[(usize, usize)]) -> Vec<(usize, usize)> {
    if runs.is_empty() {
        return Vec::new();
    }
    let max_gap = direct_dictionary_selected_window_max_gap();
    let max_amplification = direct_dictionary_selected_window_max_amplification();
    let mut windows = Vec::new();
    let mut window_start = runs[0].0;
    let mut window_end = runs[0].0 + runs[0].1;
    let mut selected_rows = runs[0].1;
    for &(run_start, run_len) in &runs[1..] {
        let run_end = run_start + run_len;
        let gap = run_start.saturating_sub(window_end);
        let merged_len = run_end - window_start;
        let merged_selected = selected_rows + run_len;
        let amplification = merged_len as f64 / merged_selected.max(1) as f64;
        if gap <= max_gap && amplification <= max_amplification {
            window_end = run_end;
            selected_rows = merged_selected;
        } else {
            windows.push((window_start, window_end - window_start));
            window_start = run_start;
            window_end = run_end;
            selected_rows = run_len;
        }
    }
    windows.push((window_start, window_end - window_start));
    windows
}

fn coalesce_selected_runs_to_i64_windows(runs: &[(usize, usize)]) -> Vec<(usize, usize)> {
    coalesce_selected_runs_to_windows_with_limits(
        runs,
        direct_dictionary_selected_i64_window_max_gap(),
        direct_dictionary_selected_i64_window_max_amplification(),
    )
}

fn coalesce_selected_runs_to_windows_with_limits(
    runs: &[(usize, usize)],
    max_gap: usize,
    max_amplification: f64,
) -> Vec<(usize, usize)> {
    if runs.is_empty() {
        return Vec::new();
    }
    let mut windows = Vec::new();
    let mut window_start = runs[0].0;
    let mut window_end = runs[0].0 + runs[0].1;
    let mut selected_rows = runs[0].1;
    for &(run_start, run_len) in &runs[1..] {
        let run_end = run_start + run_len;
        let gap = run_start.saturating_sub(window_end);
        let merged_len = run_end - window_start;
        let merged_selected = selected_rows + run_len;
        let amplification = merged_len as f64 / merged_selected.max(1) as f64;
        if gap <= max_gap && amplification <= max_amplification {
            window_end = window_end.max(run_end);
            selected_rows = merged_selected;
        } else {
            windows.push((window_start, window_end - window_start));
            window_start = run_start;
            window_end = run_end;
            selected_rows = run_len;
        }
    }
    windows.push((window_start, window_end - window_start));
    windows
}

fn append_decimal_selected_runs(
    decimals: &[i64],
    row_offset: usize,
    decimal_min: Option<i64>,
    decimal_max: Option<i64>,
    runs: &mut Vec<(usize, usize)>,
    builder: &mut SelectionRunsBuilder,
) {
    if direct_decimal_selected_runs_simd_enabled()
        && append_decimal_selected_runs_simd(
            decimals,
            row_offset,
            decimal_min,
            decimal_max,
            runs,
            builder,
        )
    {
        return;
    }
    append_decimal_selected_runs_masked(
        decimals,
        row_offset,
        decimal_min,
        decimal_max,
        runs,
        builder,
    );
}

fn append_decimal_selected_runs_masked(
    decimals: &[i64],
    row_offset: usize,
    decimal_min: Option<i64>,
    decimal_max: Option<i64>,
    runs: &mut Vec<(usize, usize)>,
    builder: &mut SelectionRunsBuilder,
) {
    let mut run_start = None;
    let mut run_len = 0usize;
    let mut row = 0usize;
    while row + 64 <= decimals.len() {
        let mut mask = 0u64;
        for lane in 0..64 {
            let decimal = decimals[row + lane];
            if decimal_min.is_none_or(|min| decimal >= min)
                && decimal_max.is_none_or(|max| decimal <= max)
            {
                mask |= 1u64 << lane;
            }
        }
        append_selected_mask_as_runs(
            row_offset + row,
            64,
            mask,
            &mut run_start,
            &mut run_len,
            runs,
            builder,
        );
        row += 64;
    }
    if row < decimals.len() {
        let len = decimals.len() - row;
        let mut mask = 0u64;
        for lane in 0..len {
            let decimal = decimals[row + lane];
            if decimal_min.is_none_or(|min| decimal >= min)
                && decimal_max.is_none_or(|max| decimal <= max)
            {
                mask |= 1u64 << lane;
            }
        }
        append_selected_mask_as_runs(
            row_offset + row,
            len,
            mask,
            &mut run_start,
            &mut run_len,
            runs,
            builder,
        );
    }
    if let Some(start) = run_start {
        builder.push_run(runs, start, run_len);
    }
}

fn append_decimal_selected_runs_simd(
    decimals: &[i64],
    row_offset: usize,
    decimal_min: Option<i64>,
    decimal_max: Option<i64>,
    runs: &mut Vec<(usize, usize)>,
    builder: &mut SelectionRunsBuilder,
) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        let (Some(min), Some(max)) = (decimal_min, decimal_max) else {
            return false;
        };
        if !std::is_x86_feature_detected!("avx2") {
            return false;
        }
        unsafe {
            append_decimal_selected_runs_avx2(decimals, row_offset, min, max, runs, builder);
        }
        true
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (
            decimals,
            row_offset,
            decimal_min,
            decimal_max,
            runs,
            builder,
        );
        false
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn append_decimal_selected_runs_avx2(
    decimals: &[i64],
    row_offset: usize,
    decimal_min: i64,
    decimal_max: i64,
    runs: &mut Vec<(usize, usize)>,
    builder: &mut SelectionRunsBuilder,
) {
    use std::arch::x86_64::{
        __m256i, _mm256_castsi256_pd, _mm256_cmpgt_epi64, _mm256_loadu_si256, _mm256_movemask_pd,
        _mm256_set1_epi64x,
    };

    let min_values = _mm256_set1_epi64x(decimal_min);
    let max_values = _mm256_set1_epi64x(decimal_max);
    let mut run_start = None;
    let mut run_len = 0usize;
    let mut row = 0usize;
    while row + 64 <= decimals.len() {
        let mut mask = 0u64;
        for chunk in 0..16 {
            let chunk_row = row + chunk * 4;
            let values =
                unsafe { _mm256_loadu_si256(decimals.as_ptr().add(chunk_row).cast::<__m256i>()) };
            let below_min = _mm256_cmpgt_epi64(min_values, values);
            let above_max = _mm256_cmpgt_epi64(values, max_values);
            let below_mask = _mm256_movemask_pd(_mm256_castsi256_pd(below_min)) as u8;
            let above_mask = _mm256_movemask_pd(_mm256_castsi256_pd(above_max)) as u8;
            let selected_mask = (!(below_mask | above_mask)) & 0x0f;
            mask |= u64::from(selected_mask) << (chunk * 4);
        }
        append_selected_mask_as_runs(
            row_offset + row,
            64,
            mask,
            &mut run_start,
            &mut run_len,
            runs,
            builder,
        );
        row += 64;
    }
    while row + 4 <= decimals.len() {
        let values = unsafe { _mm256_loadu_si256(decimals.as_ptr().add(row).cast::<__m256i>()) };
        let below_min = _mm256_cmpgt_epi64(min_values, values);
        let above_max = _mm256_cmpgt_epi64(values, max_values);
        let below_mask = _mm256_movemask_pd(_mm256_castsi256_pd(below_min)) as u8;
        let above_mask = _mm256_movemask_pd(_mm256_castsi256_pd(above_max)) as u8;
        let selected_mask = (!(below_mask | above_mask)) & 0x0f;
        append_selected_mask_as_runs(
            row_offset + row,
            4,
            u64::from(selected_mask),
            &mut run_start,
            &mut run_len,
            runs,
            builder,
        );
        row += 4;
    }
    if row < decimals.len() {
        let len = decimals.len() - row;
        let mut mask = 0u64;
        for lane in 0..len {
            let decimal = decimals[row + lane];
            if decimal >= decimal_min && decimal <= decimal_max {
                mask |= 1u64 << lane;
            }
        }
        append_selected_mask_as_runs(
            row_offset + row,
            len,
            mask,
            &mut run_start,
            &mut run_len,
            runs,
            builder,
        );
    }
    if let Some(start) = run_start {
        builder.push_run(runs, start, run_len);
    }
}

fn runs_overlap(runs: &[(usize, usize)], start: usize, end: usize) -> bool {
    runs.iter().any(|(run_start, run_len)| {
        let run_end = run_start + run_len;
        *run_start < end && run_end > start
    })
}

fn advance_run_cursor(runs: &[(usize, usize)], cursor: &mut usize, page_start: usize) {
    while let Some((run_start, run_len)) = runs.get(*cursor) {
        if run_start.saturating_add(*run_len) > page_start {
            break;
        }
        *cursor += 1;
    }
}

fn runs_overlap_from(runs: &[(usize, usize)], cursor: usize, start: usize, end: usize) -> bool {
    let Some((run_start, run_len)) = runs.get(cursor) else {
        return false;
    };
    let run_end = run_start.saturating_add(*run_len);
    *run_start < end && run_end > start
}

fn dictionary_fallback_id(dictionary: &mut Vec<Bytes>, fallback: &[u8]) -> Result<i32> {
    if let Some(index) = dictionary
        .iter()
        .position(|candidate| candidate.as_ref() == fallback)
    {
        return i32::try_from(index).map_err(|_| {
            DodamError::UnsupportedSql("dictionary fallback id out of range".to_string())
        });
    }
    let id = i32::try_from(dictionary.len()).map_err(|_| {
        DodamError::UnsupportedSql("dictionary fallback id out of range".to_string())
    })?;
    dictionary.push(Bytes::copy_from_slice(fallback));
    Ok(id)
}

fn compact_selected_dictionary_ids(
    def_levels: &[i16],
    ids: &[i32],
    row_offset: usize,
    records: usize,
    value_offset: usize,
    fallback_id: i32,
    runs: &[(usize, usize)],
    output: &mut Vec<i32>,
) -> Result<()> {
    if def_levels.is_empty() {
        for &(start, len) in runs {
            let value_start = value_offset + start;
            let value_end = value_start + len;
            if value_end > ids.len() {
                return Err(DodamError::UnsupportedSql(
                    "dictionary id length mismatch".to_string(),
                ));
            }
            output.extend_from_slice(&ids[value_start..value_end]);
        }
        return Ok(());
    }
    if row_offset + records > def_levels.len() {
        return Err(DodamError::UnsupportedSql(
            "dictionary definition level length mismatch".to_string(),
        ));
    }
    let mut value_index = value_offset;
    let mut next_run = 0usize;
    for row in 0..records {
        let selected = next_run < runs.len()
            && row >= runs[next_run].0
            && row < runs[next_run].0 + runs[next_run].1;
        let present = def_levels[row_offset + row] != 0;
        if selected {
            if present {
                let Some(id) = ids.get(value_index).copied() else {
                    return Err(DodamError::UnsupportedSql(
                        "dictionary id length mismatch".to_string(),
                    ));
                };
                output.push(id);
            } else {
                output.push(fallback_id);
            }
        }
        if present {
            value_index += 1;
        }
        if next_run < runs.len() && row + 1 == runs[next_run].0 + runs[next_run].1 {
            next_run += 1;
        }
    }
    Ok(())
}

fn advance_dictionary_value_offset(
    def_levels: &[i16],
    row_offset: usize,
    records: usize,
    value_offset: usize,
) -> usize {
    if def_levels.is_empty() {
        value_offset + records
    } else {
        value_offset
            + def_levels[row_offset..row_offset + records]
                .iter()
                .filter(|level| **level != 0)
                .count()
    }
}

fn direct_selection_payload_gate(
    records: usize,
    selected_rows: usize,
    selected_runs: usize,
) -> bool {
    let max_ratio = std::env::var("DODAM_DIRECT_SELECTION_MAX_RATIO")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(0.20);
    let min_run_len = std::env::var("DODAM_DIRECT_SELECTION_MIN_RUN_LEN")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(32);
    let cached_min_run_len = std::env::var("DODAM_DIRECT_SELECTION_CACHED_MIN_RUN_LEN")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(8);
    let cached_reread = std::env::var("DODAM_DIRECT_SELECTION_CACHED_REREAD")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"));
    let decision = choose_selected_payload(SelectedPayloadCostInput {
        records,
        selected_rows,
        selected_runs,
        max_selected_ratio: max_ratio,
        min_average_run_len: min_run_len,
        cached_reread,
        cached_min_average_run_len: cached_min_run_len,
    });
    log_direct_selection_gate(
        records,
        selected_rows,
        selected_runs,
        decision.accepted(),
        decision.reason(),
    );
    decision.accepted()
}

fn direct_dictionary_selected_full_payload_enabled() -> bool {
    std::env::var("DODAM_FUSED_DICT_SELECTED_ALLOW_FULL_PAYLOAD")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn direct_dictionary_selected_page_decode_enabled() -> bool {
    !std::env::var("DODAM_DISABLE_FUSED_DICT_SELECTED_PAGE_DECODE")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn direct_dictionary_selected_full_primitive_payload_enabled() -> bool {
    std::env::var("DODAM_FUSED_DICT_SELECTED_FULL_PRIMITIVE_PAYLOAD")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn direct_dictionary_selected_masked_full_payload_enabled() -> bool {
    std::env::var("DODAM_FUSED_DICT_SELECTED_MASKED_FULL_PAYLOAD")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn direct_dictionary_selected_primitive_page_slice_enabled() -> bool {
    std::env::var("DODAM_FUSED_DICT_SELECTED_PRIMITIVE_PAGE_SLICE")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        || direct_fused_selected_i64_page_decoder_enabled()
}

fn direct_fused_selected_i64_page_decoder_enabled() -> bool {
    std::env::var("DODAM_ENABLE_FUSED_SELECTED_I64_PAGE_DECODER")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn direct_fused_selected_i64_aggregate_sink_enabled() -> bool {
    std::env::var("DODAM_ENABLE_FUSED_SELECTED_I64_AGG_SINK")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn direct_dictionary_decimal_selected_runs_enabled() -> bool {
    std::env::var("DODAM_ENABLE_DICTIONARY_DECIMAL_SELECTED_RUNS")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn direct_decimal_selected_runs_simd_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        !std::env::var("DODAM_DISABLE_DECIMAL_SELECTED_RUNS_SIMD")
            .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
    })
}

fn direct_dictionary_selected_window_payload_enabled() -> bool {
    std::env::var("DODAM_FUSED_DICT_SELECTED_WINDOW_PAYLOAD")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn direct_dictionary_selected_i64_window_payload_enabled() -> bool {
    std::env::var("DODAM_ENABLE_FUSED_SELECTED_I64_WINDOW_READER")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn direct_dictionary_selected_window_max_gap() -> usize {
    std::env::var("DODAM_FUSED_DICT_SELECTED_WINDOW_MAX_GAP")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(512)
}

fn direct_dictionary_selected_window_max_amplification() -> f64 {
    std::env::var("DODAM_FUSED_DICT_SELECTED_WINDOW_MAX_AMPLIFICATION")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value >= 1.0)
        .unwrap_or(4.0)
}

fn direct_dictionary_selected_i64_window_max_gap() -> usize {
    std::env::var("DODAM_FUSED_SELECTED_I64_WINDOW_MAX_GAP")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(2048)
}

fn direct_dictionary_selected_i64_window_max_amplification() -> f64 {
    std::env::var("DODAM_FUSED_SELECTED_I64_WINDOW_MAX_AMPLIFICATION")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value >= 1.0)
        .unwrap_or(2.0)
}

fn log_direct_selection_gate(
    records: usize,
    selected_rows: usize,
    selected_runs: usize,
    accepted: bool,
    reason: &str,
) {
    if !std::env::var("DODAM_DIRECT_SELECTION_TRACE")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
    {
        return;
    }
    let ratio = if records == 0 {
        0.0
    } else {
        selected_rows as f64 / records as f64
    };
    let average_run_len = if selected_runs == 0 {
        0.0
    } else {
        selected_rows as f64 / selected_runs as f64
    };
    eprintln!(
        "[dodam:direct-selection] accepted={} reason={} records={} selected={} ratio={:.6} runs={} avg_run_len={:.3}",
        accepted, reason, records, selected_rows, ratio, selected_runs, average_run_len
    );
}

fn direct_def_levels_match(level_count: usize, record_count: usize, required: bool) -> bool {
    if required {
        level_count == 0
    } else {
        level_count == record_count
    }
}

fn direct_all_present(required: bool, def_levels: &[i16]) -> bool {
    required || def_levels.iter().all(|level| *level != 0)
}

fn direct_value_count_matches(value_count: usize, record_count: usize, required: bool) -> bool {
    if required {
        value_count == record_count
    } else {
        value_count <= record_count
    }
}

fn direct_skip_required_def_levels_enabled() -> bool {
    std::env::var("DODAM_DIRECT_SKIP_REQUIRED_DEF_LEVELS")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn direct_primitive_page_reader_enabled() -> bool {
    std::env::var("DODAM_ENABLE_PRIMITIVE_PAGE_READER")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

enum I64SetPredicate {
    Dense(Vec<bool>),
    Hash(HashSet<i64>),
}

impl I64SetPredicate {
    const MAX_DENSE_KEY: usize = 20_000_000;

    fn from_hash_set(keys: &HashSet<i64>) -> Self {
        let mut max_key = 0_usize;
        for key in keys.iter().copied() {
            let Ok(index) = usize::try_from(key) else {
                return Self::Hash(keys.clone());
            };
            if index > Self::MAX_DENSE_KEY {
                return Self::Hash(keys.clone());
            }
            max_key = max_key.max(index);
        }
        let mut dense = vec![false; max_key.saturating_add(1)];
        for key in keys.iter().copied() {
            let index = usize::try_from(key).expect("validated dense i64 set key");
            dense[index] = true;
        }
        Self::Dense(dense)
    }

    fn contains(&self, key: i64) -> bool {
        match self {
            Self::Dense(values) => {
                let Ok(index) = usize::try_from(key) else {
                    return false;
                };
                values.get(index).copied().unwrap_or(false)
            }
            Self::Hash(values) => values.contains(&key),
        }
    }

    fn evaluate(&self, values: &Int64Array) -> BooleanArray {
        let mut output = BooleanBuilder::with_capacity(values.len());
        if values.null_count() == 0 {
            for row in 0..values.len() {
                output.append_value(self.contains(values.value(row)));
            }
        } else {
            for row in 0..values.len() {
                output.append_value(!values.is_null(row) && self.contains(values.value(row)));
            }
        }
        output.finish()
    }
}

pub struct I64BloomPredicate {
    bits: Vec<u64>,
    mask: u64,
    min_key: i64,
    max_key: i64,
}

impl I64BloomPredicate {
    pub(crate) fn from_hash_set(keys: &HashSet<i64>) -> Self {
        let expected_rows = keys.len().max(1);
        let bit_count = expected_rows.saturating_mul(64).next_power_of_two().max(64);
        let mut bloom = Self {
            bits: vec![0; bit_count.div_ceil(64)],
            mask: (bit_count - 1) as u64,
            min_key: i64::MAX,
            max_key: i64::MIN,
        };
        for key in keys.iter().copied() {
            bloom.min_key = bloom.min_key.min(key);
            bloom.max_key = bloom.max_key.max(key);
            bloom.insert(key);
        }
        bloom
    }

    fn insert(&mut self, key: i64) {
        for hash in i64_bloom_hashes(key) {
            self.set(hash);
        }
    }

    fn might_contain(&self, key: i64) -> bool {
        key >= self.min_key
            && key <= self.max_key
            && i64_bloom_hashes(key).into_iter().all(|hash| self.get(hash))
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

    fn evaluate(&self, values: &Int64Array) -> BooleanArray {
        let mut output = BooleanBuilder::with_capacity(values.len());
        if values.null_count() == 0 {
            for row in 0..values.len() {
                output.append_value(self.might_contain(values.value(row)));
            }
        } else {
            for row in 0..values.len() {
                output.append_value(!values.is_null(row) && self.might_contain(values.value(row)));
            }
        }
        output.finish()
    }
}

fn i64_bloom_hashes(value: i64) -> [u64; 3] {
    let hash = splitmix64(value as u64);
    [
        hash,
        splitmix64(hash ^ 0x9e37_79b9_7f4a_7c15),
        splitmix64(hash ^ 0xc2b2_ae3d_27d4_eb4f),
    ]
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
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
    maybe_profile_parquet_projected_columns(
        path,
        &builder,
        &column_indices,
        &all_row_groups,
        &row_groups,
    );

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

pub fn read_parquet_projection_compressed_bytes(
    path: impl AsRef<Path>,
    projection: &Projection,
    metadata_cache: &ParquetMetadataCache,
    store: &dyn ObjectStore,
) -> Result<u64> {
    let path = path.as_ref();
    let file = store.open(path)?;
    let metadata = metadata_cache.get_with_store(path, store)?;
    let builder = ParquetRecordBatchReaderBuilder::new_with_metadata(file, metadata);
    let row_groups = builder.metadata().num_row_groups();
    let all_row_groups = (0..row_groups).collect::<Vec<_>>();
    let column_indices = projection_indices_for_schema(builder.schema(), projection)?;
    Ok(compressed_bytes_for_row_groups(
        &builder,
        &column_indices,
        &all_row_groups,
    ))
}

pub fn read_parquet_i64_column_max(
    path: impl AsRef<Path>,
    column: &str,
    metadata_cache: &ParquetMetadataCache,
    store: &dyn ObjectStore,
) -> Result<Option<i64>> {
    let path = path.as_ref();
    let file = store.open(path)?;
    let metadata = metadata_cache.get_with_store(path, store)?;
    let builder = ParquetRecordBatchReaderBuilder::new_with_metadata(file, metadata);
    let Some(column_index) = builder
        .schema()
        .fields()
        .iter()
        .position(|field| field.name() == column && field.data_type() == &DataType::Int64)
    else {
        return Ok(None);
    };
    let mut max_value = None;
    for row_group in builder.metadata().row_groups() {
        let Some(column) = row_group.columns().get(column_index) else {
            return Ok(None);
        };
        let Some(statistics) = column.statistics() else {
            return Ok(None);
        };
        if statistics.is_min_max_deprecated()
            || !statistics.min_is_exact()
            || !statistics.max_is_exact()
        {
            return Ok(None);
        }
        let Statistics::Int64(statistics) = statistics else {
            return Ok(None);
        };
        let Some(value) = statistics.max_opt().copied() else {
            return Ok(None);
        };
        max_value = Some(max_value.map_or(value, |current: i64| current.max(value)));
    }
    Ok(max_value)
}

pub fn read_parquet_primitive_column_min_max_by_row_group(
    path: impl AsRef<Path>,
    column: &str,
    metadata_cache: &ParquetMetadataCache,
    store: &dyn ObjectStore,
) -> Result<Option<Vec<PrimitiveRowGroupMinMax>>> {
    let path = path.as_ref();
    let metadata = metadata_cache.get_with_store(path, store)?;
    let Some(column_index) = metadata.schema().fields().iter().position(|field| {
        field.name() == column
            && matches!(
                field.data_type(),
                DataType::Int32
                    | DataType::Int64
                    | DataType::Date32
                    | DataType::Date64
                    | DataType::Timestamp(_, _)
                    | DataType::UInt32
                    | DataType::Decimal128(_, _)
            )
    }) else {
        return Ok(None);
    };
    let column_type = metadata.schema().field(column_index).data_type().clone();
    let parquet_metadata = metadata.metadata();
    let mut ranges = Vec::with_capacity(parquet_metadata.num_row_groups());
    for (row_group_index, row_group) in parquet_metadata.row_groups().iter().enumerate() {
        let Some(column) = row_group.columns().get(column_index) else {
            return Ok(None);
        };
        let Some(statistics) = column.statistics() else {
            return Ok(None);
        };
        if statistics.is_min_max_deprecated()
            || !statistics.min_is_exact()
            || !statistics.max_is_exact()
        {
            return Ok(None);
        }
        let Some((min, max)) = row_group_i128_min_max(&column_type, statistics) else {
            return Ok(None);
        };
        ranges.push(PrimitiveRowGroupMinMax {
            row_group: row_group_index,
            rows: usize::try_from(row_group.num_rows()).unwrap_or(usize::MAX),
            null_count: statistics.null_count_opt(),
            min,
            max,
        });
    }
    Ok(Some(ranges))
}

pub fn read_parquet_i32_column_min_max_for_row_groups_with_store(
    path: &Path,
    column: &str,
    row_groups: &[usize],
    file_cache: Arc<ParquetFileCache>,
    store: &dyn ObjectStore,
) -> Result<Option<(i32, i32)>> {
    if file_cache.enabled() {
        let reader = CachedParquetChunkReader::new(path, store, file_cache)?;
        let reader = SerializedFileReader::new(reader)?;
        return read_parquet_i32_column_min_max_for_row_groups_reader(reader, column, row_groups);
    } else {
        let file = store.open(path)?;
        let reader = SerializedFileReader::new(file)?;
        return read_parquet_i32_column_min_max_for_row_groups_reader(reader, column, row_groups);
    }
}

fn read_parquet_i32_column_min_max_for_row_groups_reader<R: ChunkReader + 'static>(
    reader: SerializedFileReader<R>,
    column: &str,
    row_groups: &[usize],
) -> Result<Option<(i32, i32)>> {
    let Some(column_indices) = parquet_column_indices_by_name(&reader, &[column]) else {
        return Ok(None);
    };
    let [column_index] = <[usize; 1]>::try_from(column_indices).map_err(|_| {
        DodamError::UnsupportedSql("parquet i32 range column shape mismatch".to_string())
    })?;
    let schema = reader.metadata().file_metadata().schema_descr();
    let column_desc = schema.column(column_index);
    if column_desc.physical_type() != ParquetPhysicalType::INT32
        || column_desc.max_rep_level() != 0
        || column_desc.max_def_level() != 0
    {
        return Ok(None);
    }
    let mut min_value = i32::MAX;
    let mut max_value = i32::MIN;
    let mut seen = false;
    for &row_group_index in row_groups {
        let Some(row_group) = reader.metadata().row_groups().get(row_group_index) else {
            return Ok(None);
        };
        let Some(column_chunk) = row_group.columns().get(column_index) else {
            return Ok(None);
        };
        let Some(statistics) = column_chunk.statistics() else {
            return Ok(None);
        };
        if statistics.is_min_max_deprecated() {
            return Ok(None);
        }
        let Statistics::Int32(statistics) = statistics else {
            return Ok(None);
        };
        let (Some(min), Some(max)) = (statistics.min_opt().copied(), statistics.max_opt().copied())
        else {
            return Ok(None);
        };
        min_value = min_value.min(min);
        max_value = max_value.max(max);
        seen = true;
    }
    Ok(seen.then_some((min_value, max_value)))
}

pub fn read_parquet_i64_column_constant(
    path: impl AsRef<Path>,
    column: &str,
    metadata_cache: &ParquetMetadataCache,
    store: &dyn ObjectStore,
) -> Result<Option<i64>> {
    let path = path.as_ref();
    let file = store.open(path)?;
    let metadata = metadata_cache.get_with_store(path, store)?;
    let builder = ParquetRecordBatchReaderBuilder::new_with_metadata(file, metadata);
    let Some(column_index) = builder.schema().fields().iter().position(|field| {
        field.name() == column && matches!(field.data_type(), DataType::Int32 | DataType::Int64)
    }) else {
        return Ok(None);
    };
    let column_nullable = builder.schema().field(column_index).is_nullable();
    let mut constant = None;
    for row_group in builder.metadata().row_groups() {
        let Some(column) = row_group.columns().get(column_index) else {
            return Ok(None);
        };
        let Some(statistics) = column.statistics() else {
            return Ok(None);
        };
        if statistics.is_min_max_deprecated()
            || !statistics.min_is_exact()
            || !statistics.max_is_exact()
            || (column_nullable && statistics.null_count_opt() != Some(0))
        {
            return Ok(None);
        }
        let (min_value, max_value) = match statistics {
            Statistics::Int64(statistics) => {
                let (Some(min_value), Some(max_value)) =
                    (statistics.min_opt().copied(), statistics.max_opt().copied())
                else {
                    return Ok(None);
                };
                (min_value, max_value)
            }
            Statistics::Int32(statistics) => {
                let (Some(min_value), Some(max_value)) =
                    (statistics.min_opt().copied(), statistics.max_opt().copied())
                else {
                    return Ok(None);
                };
                (i64::from(min_value), i64::from(max_value))
            }
            _ => return Ok(None),
        };
        if min_value != max_value {
            return Ok(None);
        }
        match constant {
            Some(value) if value != min_value => return Ok(None),
            Some(_) => {}
            None => constant = Some(min_value),
        }
    }
    Ok(constant)
}

pub fn read_parquet_i128_column_min_max(
    path: impl AsRef<Path>,
    column: &str,
    metadata_cache: &ParquetMetadataCache,
    store: &dyn ObjectStore,
) -> Result<Option<(i128, i128)>> {
    let path = path.as_ref();
    let metadata = metadata_cache.get_with_store(path, store)?;
    let Some(column_index) = metadata
        .schema()
        .fields()
        .iter()
        .position(|field| field.name() == column)
    else {
        return Ok(None);
    };
    let data_type = metadata.schema().field(column_index).data_type();
    let mut min_value = i128::MAX;
    let mut max_value = i128::MIN;
    let mut seen = false;
    for row_group in metadata.metadata().row_groups() {
        let Some(column_chunk) = row_group.columns().get(column_index) else {
            return Ok(None);
        };
        let Some(statistics) = column_chunk.statistics() else {
            return Ok(None);
        };
        if statistics.is_min_max_deprecated()
            || !statistics.min_is_exact()
            || !statistics.max_is_exact()
        {
            return Ok(None);
        }
        let Some((min, max)) = row_group_i128_min_max(data_type, statistics) else {
            return Ok(None);
        };
        min_value = min_value.min(min);
        max_value = max_value.max(max);
        seen = true;
    }
    Ok(seen.then_some((min_value, max_value)))
}

pub fn read_parquet_i128_column_min_max_relaxed(
    path: impl AsRef<Path>,
    column: &str,
    metadata_cache: &ParquetMetadataCache,
    store: &dyn ObjectStore,
) -> Result<Option<(i128, i128)>> {
    let path = path.as_ref();
    let metadata = metadata_cache.get_with_store(path, store)?;
    let Some(column_index) = metadata
        .schema()
        .fields()
        .iter()
        .position(|field| field.name() == column)
    else {
        return Ok(None);
    };
    let data_type = metadata.schema().field(column_index).data_type();
    let mut min_value = i128::MAX;
    let mut max_value = i128::MIN;
    let mut seen = false;
    for row_group in metadata.metadata().row_groups() {
        let Some(column_chunk) = row_group.columns().get(column_index) else {
            return Ok(None);
        };
        let Some(statistics) = column_chunk.statistics() else {
            return Ok(None);
        };
        let Some((min, max)) = row_group_i128_min_max(data_type, statistics) else {
            return Ok(None);
        };
        min_value = min_value.min(min);
        max_value = max_value.max(max);
        seen = true;
    }
    Ok(seen.then_some((min_value, max_value)))
}

pub fn parquet_row_groups_monotonic_by_column(
    path: impl AsRef<Path>,
    column: &str,
    metadata_cache: &ParquetMetadataCache,
    store: &dyn ObjectStore,
) -> Result<bool> {
    let path = path.as_ref();
    let file = store.open(path)?;
    let metadata = metadata_cache.get_with_store(path, store)?;
    let builder = ParquetRecordBatchReaderBuilder::new_with_metadata(file, metadata);
    let Some(column_index) = builder
        .schema()
        .fields()
        .iter()
        .position(|field| field.name() == column)
    else {
        return Ok(false);
    };
    let data_type = builder.schema().field(column_index).data_type();
    let mut previous_max = None::<i128>;
    for row_group in builder.metadata().row_groups() {
        let Some(column_chunk) = row_group.columns().get(column_index) else {
            return Ok(false);
        };
        let Some(statistics) = column_chunk.statistics() else {
            return Ok(false);
        };
        if statistics.is_min_max_deprecated()
            || !statistics.min_is_exact()
            || !statistics.max_is_exact()
            || statistics.null_count_opt().is_some_and(|nulls| nulls > 0)
        {
            return Ok(false);
        }
        let Some((min_value, max_value)) = row_group_i128_min_max(data_type, statistics) else {
            return Ok(false);
        };
        if let Some(previous_max) = previous_max
            && min_value < previous_max
        {
            return Ok(false);
        }
        previous_max = Some(max_value);
    }
    Ok(previous_max.is_some())
}

pub fn parquet_column_monotonic_by_scan(
    path: impl AsRef<Path>,
    column: &str,
    batch_size: usize,
    metadata_cache: &ParquetMetadataCache,
    store: &dyn ObjectStore,
) -> Result<bool> {
    let path = path.as_ref();
    let file = store.open(path)?;
    let metadata = metadata_cache.get_with_store(path, store)?;
    let builder = ParquetRecordBatchReaderBuilder::new_with_metadata(file, metadata);
    let Some(column_index) = builder
        .schema()
        .fields()
        .iter()
        .position(|field| field.name() == column)
    else {
        return Ok(false);
    };
    let data_type = builder.schema().field(column_index).data_type().clone();
    let mask = ProjectionMask::roots(builder.parquet_schema(), [column_index]);
    let mut reader = builder
        .with_projection(mask)
        .with_batch_size(batch_size)
        .build()?;
    let mut previous = None::<i128>;
    let mut saw_value = false;
    for batch in &mut reader {
        let batch = batch?;
        let Some(array) = batch
            .column(0)
            .as_ref()
            .as_any()
            .downcast_ref::<Int32Array>()
        else {
            if let Some(array) = batch
                .column(0)
                .as_ref()
                .as_any()
                .downcast_ref::<Int64Array>()
            {
                for row in 0..array.len() {
                    if array.is_null(row) {
                        return Ok(false);
                    }
                    let value = i128::from(array.value(row));
                    if previous.is_some_and(|previous| value < previous) {
                        return Ok(false);
                    }
                    previous = Some(value);
                    saw_value = true;
                }
                continue;
            }
            return Ok(false);
        };
        if !matches!(data_type, DataType::Int32 | DataType::Date32) {
            return Ok(false);
        }
        for row in 0..array.len() {
            if array.is_null(row) {
                return Ok(false);
            }
            let value = i128::from(array.value(row));
            if previous.is_some_and(|previous| value < previous) {
                return Ok(false);
            }
            previous = Some(value);
            saw_value = true;
        }
    }
    Ok(saw_value)
}

fn row_group_i128_min_max(data_type: &DataType, statistics: &Statistics) -> Option<(i128, i128)> {
    match (data_type, statistics) {
        (DataType::Int32 | DataType::Date32, Statistics::Int32(typed)) => Some((
            i128::from(typed.min_opt().copied()?),
            i128::from(typed.max_opt().copied()?),
        )),
        (
            DataType::Int64 | DataType::Date64 | DataType::Timestamp(_, _),
            Statistics::Int64(typed),
        ) => Some((
            i128::from(typed.min_opt().copied()?),
            i128::from(typed.max_opt().copied()?),
        )),
        (DataType::UInt32, Statistics::Int32(typed)) => Some((
            i128::from(u32::try_from(typed.min_opt().copied()?).ok()?),
            i128::from(u32::try_from(typed.max_opt().copied()?).ok()?),
        )),
        (DataType::Decimal128(_, _), Statistics::Int32(typed)) => Some((
            i128::from(typed.min_opt().copied()?),
            i128::from(typed.max_opt().copied()?),
        )),
        (DataType::Decimal128(_, _), Statistics::Int64(typed)) => Some((
            i128::from(typed.min_opt().copied()?),
            i128::from(typed.max_opt().copied()?),
        )),
        (DataType::Decimal128(_, _), Statistics::FixedLenByteArray(typed)) => Some((
            fixed_len_decimal_to_i128(typed.min_opt()?)?,
            fixed_len_decimal_to_i128(typed.max_opt()?)?,
        )),
        (DataType::Decimal128(_, _), Statistics::ByteArray(typed)) => Some((
            typed
                .min_opt()
                .and_then(|value| decimal_bytes_to_i128(value.as_ref()))?,
            typed
                .max_opt()
                .and_then(|value| decimal_bytes_to_i128(value.as_ref()))?,
        )),
        _ => None,
    }
}

impl Iterator for ParquetBatchReader {
    type Item = Result<RecordBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        let started = Instant::now();
        let next = self.inner.next();
        let next_nanos = elapsed_nanos(started);
        self.next_calls = self.next_calls.saturating_add(1);
        self.next_nanos = self.next_nanos.saturating_add(next_nanos);
        self.max_next_nanos = self.max_next_nanos.max(next_nanos);
        if let Some(samples) = self.next_samples.as_mut() {
            samples.push(next_nanos);
        }
        match next {
            Some(Ok(batch)) => {
                let batch = match enforce_projection_order_cached(batch, &mut self.projection_order)
                {
                    Ok(batch) => batch,
                    Err(error) => return Some(Err(error)),
                };
                self.output_batches = self.output_batches.saturating_add(1);
                self.output_rows = self.output_rows.saturating_add(batch.num_rows());
                if batch.num_rows() == 0 {
                    self.zero_row_batches = self.zero_row_batches.saturating_add(1);
                }
                Some(Ok(batch))
            }
            Some(Err(error)) => Some(Err(error.into())),
            None => {
                self.eof_calls = self.eof_calls.saturating_add(1);
                None
            }
        }
    }
}

fn apply_projection<T: ChunkReader + 'static>(
    builder: ParquetRecordBatchReaderBuilder<T>,
    projection: &Projection,
) -> Result<ParquetRecordBatchReaderBuilder<T>> {
    let Projection::Columns(columns) = projection else {
        return Ok(builder);
    };

    let indices = projection_indices(builder.schema(), columns)?;
    let mask = ProjectionMask::roots(builder.parquet_schema(), indices);
    Ok(builder.with_projection(mask))
}

fn projection_order(projection: &Projection) -> Option<ProjectionOrderState> {
    match projection {
        Projection::All => None,
        Projection::Columns(columns) => Some(ProjectionOrderState::Pending(columns.clone())),
    }
}

fn enforce_projection_order_cached(
    batch: RecordBatch,
    projection_order: &mut Option<ProjectionOrderState>,
) -> Result<RecordBatch> {
    let Some(projection_order) = projection_order else {
        return Ok(batch);
    };
    if let ProjectionOrderState::Pending(columns) = projection_order {
        let reorder = projection_reorder_indices(&batch, columns)?;
        *projection_order = ProjectionOrderState::Ready(reorder);
    }
    let ProjectionOrderState::Ready(Some(indices)) = projection_order else {
        return Ok(batch);
    };
    reorder_record_batch(batch, indices)
}

fn projection_reorder_indices(
    batch: &RecordBatch,
    projection_order: &[String],
) -> Result<Option<Vec<usize>>> {
    if projection_order.len() != batch.num_columns() {
        return Ok(None);
    }
    let mut indices = Vec::with_capacity(projection_order.len());
    for column in projection_order {
        indices.push(schema_column_index(&batch.schema(), column)?);
    }
    if indices.iter().copied().eq(0..indices.len()) {
        Ok(None)
    } else {
        Ok(Some(indices))
    }
}

fn reorder_record_batch(batch: RecordBatch, indices: &[usize]) -> Result<RecordBatch> {
    let mut columns = Vec::with_capacity(indices.len());
    let mut fields = Vec::with_capacity(indices.len());
    for &index in indices {
        columns.push(batch.column(index).clone());
        fields.push(batch.schema().field(index).clone());
    }
    Ok(RecordBatch::try_new(
        Arc::new(Schema::new(fields)),
        columns,
    )?)
}

fn row_filter<T: ChunkReader + 'static>(
    builder: &ParquetRecordBatchReaderBuilder<T>,
    predicates: &[Expr],
) -> Result<Option<RowFilter>> {
    if predicates.is_empty() {
        return Ok(None);
    }
    let mut arrow_predicates = Vec::<Box<dyn ArrowPredicate>>::new();
    for predicate in predicates {
        let mut columns = Vec::new();
        collect_expr_columns(predicate, &mut columns);
        if columns.is_empty() {
            continue;
        }
        let indices = projection_indices(builder.schema(), &columns)?;
        let mask = ProjectionMask::roots(builder.parquet_schema(), indices);
        let filter = FilterExpr::new(predicate.clone());
        arrow_predicates.push(Box::new(ArrowPredicateFn::new(mask, move |batch| {
            evaluate_filter_mask(&batch, &filter)
                .map_err(|error| ArrowError::ComputeError(error.to_string()))
        })));
    }
    if arrow_predicates.is_empty() {
        Ok(None)
    } else {
        Ok(Some(RowFilter::new(arrow_predicates)))
    }
}

fn collect_expr_columns(expr: &Expr, columns: &mut Vec<String>) {
    match expr {
        Expr::Boolean(_) => {}
        Expr::Comparison(comparison) => push_unique_column(columns, &comparison.column),
        Expr::ColumnComparison { left, right, .. } => {
            push_unique_column(columns, left);
            push_unique_column(columns, right);
        }
        Expr::InList { column, .. } | Expr::Like { column, .. } | Expr::IsNull { column, .. } => {
            push_unique_column(columns, column);
        }
        Expr::Not(expr) => collect_expr_columns(expr, columns),
        Expr::And(left, right) | Expr::Or(left, right) => {
            collect_expr_columns(left, columns);
            collect_expr_columns(right, columns);
        }
    }
}

fn push_unique_column(columns: &mut Vec<String>, column: &str) {
    if !columns.iter().any(|existing| existing == column) {
        columns.push(column.to_string());
    }
}

fn projected_column_count(schema: &arrow::datatypes::SchemaRef, projection: &Projection) -> usize {
    match projection {
        Projection::All => schema.fields().len(),
        Projection::Columns(columns) => columns.len(),
    }
}

fn metadata_with_dictionary_columns(
    metadata: ArrowReaderMetadata,
    dictionary_columns: &[String],
) -> Result<ArrowReaderMetadata> {
    if dictionary_columns.is_empty() {
        return Ok(metadata);
    }
    let dictionary_columns = dictionary_columns
        .iter()
        .map(|column| column.as_str())
        .collect::<BTreeSet<_>>();
    let fields = metadata
        .schema()
        .fields()
        .iter()
        .map(|field| {
            if dictionary_columns.contains(field.name().as_str())
                && matches!(field.data_type(), DataType::Utf8 | DataType::LargeUtf8)
            {
                Arc::new(Field::new(
                    field.name(),
                    DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
                    field.is_nullable(),
                ))
            } else {
                field.clone()
            }
        })
        .collect::<Vec<_>>();
    let schema = Arc::new(Schema::new(fields));
    let options = arrow_reader_options().with_schema(schema);
    Ok(ArrowReaderMetadata::try_new(
        metadata.metadata().clone(),
        options,
    )?)
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
        .map(|column| schema_column_index(schema, column))
        .collect()
}

fn schema_column_index(schema: &arrow::datatypes::SchemaRef, column: &str) -> Result<usize> {
    if let Some(index) = schema
        .fields()
        .iter()
        .position(|field| field.name() == column)
    {
        return Ok(index);
    }
    if let Some((_, unqualified)) = column.split_once('.')
        && let Some(index) = schema
            .fields()
            .iter()
            .position(|field| field.name() == unqualified)
    {
        return Ok(index);
    }
    Err(DodamError::UnknownColumn(column.to_string()))
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

fn compressed_bytes_for_row_groups<T: ChunkReader + 'static>(
    builder: &ParquetRecordBatchReaderBuilder<T>,
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

fn maybe_profile_parquet_projected_columns<T: ChunkReader + 'static>(
    path: &Path,
    builder: &ParquetRecordBatchReaderBuilder<T>,
    column_indices: &[usize],
    all_row_groups: &[usize],
    scanned_row_groups: &[usize],
) {
    if !parquet_column_profile_enabled() {
        return;
    }
    let fields = builder.schema().fields();
    let columns = column_indices
        .iter()
        .map(|column_index| {
            let name = fields
                .get(*column_index)
                .map(|field| field.name().as_str())
                .unwrap_or("<unknown>");
            let total_compressed =
                parquet_column_compressed_bytes(builder, *column_index, all_row_groups);
            let scanned_compressed =
                parquet_column_compressed_bytes(builder, *column_index, scanned_row_groups);
            let scanned_uncompressed =
                parquet_column_uncompressed_bytes(builder, *column_index, scanned_row_groups);
            let encodings = parquet_column_encodings(builder, *column_index, scanned_row_groups);
            format!(
                "{name}:compressed={scanned_compressed}/{total_compressed} uncompressed={scanned_uncompressed} encodings={}",
                encodings.into_iter().collect::<Vec<_>>().join("|")
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    eprintln!(
        "[dodam:parquet-column-profile] {}: row_groups={}/{} columns=[{}]",
        path.display(),
        scanned_row_groups.len(),
        all_row_groups.len(),
        columns
    );
}

fn parquet_column_profile_enabled() -> bool {
    std::env::var("DODAM_PARQUET_COLUMN_PROFILE")
        .or_else(|_| std::env::var("DODAM_SCAN_PROFILE"))
        .or_else(|_| std::env::var("DODAM_TPCH_PROFILE"))
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn parquet_column_chunk_profile_enabled() -> bool {
    std::env::var("DODAM_PARQUET_COLUMN_CHUNK_PROFILE")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn parquet_column_compressed_bytes<T: ChunkReader + 'static>(
    builder: &ParquetRecordBatchReaderBuilder<T>,
    column_index: usize,
    row_groups: &[usize],
) -> u64 {
    row_groups
        .iter()
        .filter_map(|row_group| builder.metadata().row_groups().get(*row_group))
        .filter_map(|row_group| row_group.columns().get(column_index))
        .map(|column| column.compressed_size().max(0) as u64)
        .sum()
}

fn parquet_column_uncompressed_bytes<T: ChunkReader + 'static>(
    builder: &ParquetRecordBatchReaderBuilder<T>,
    column_index: usize,
    row_groups: &[usize],
) -> u64 {
    row_groups
        .iter()
        .filter_map(|row_group| builder.metadata().row_groups().get(*row_group))
        .filter_map(|row_group| row_group.columns().get(column_index))
        .map(|column| column.uncompressed_size().max(0) as u64)
        .sum()
}

fn parquet_column_encodings<T: ChunkReader + 'static>(
    builder: &ParquetRecordBatchReaderBuilder<T>,
    column_index: usize,
    row_groups: &[usize],
) -> BTreeSet<String> {
    row_groups
        .iter()
        .filter_map(|row_group| builder.metadata().row_groups().get(*row_group))
        .filter_map(|row_group| row_group.columns().get(column_index))
        .flat_map(|column| column.encodings())
        .map(|encoding| format!("{encoding:?}"))
        .collect()
}

fn prune_row_groups<T: ChunkReader + 'static>(
    builder: &ParquetRecordBatchReaderBuilder<T>,
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

fn row_group_may_match<T: ChunkReader + 'static>(
    builder: &ParquetRecordBatchReaderBuilder<T>,
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
        Expr::Like {
            column,
            pattern,
            negated,
            escape,
            case_insensitive,
        } => {
            if *case_insensitive {
                Ok(true)
            } else {
                row_group_may_match_like(
                    builder,
                    row_group_index,
                    column,
                    pattern,
                    *negated,
                    *escape,
                )
            }
        }
        Expr::Boolean(_)
        | Expr::ColumnComparison { .. }
        | Expr::InList { .. }
        | Expr::IsNull { .. }
        | Expr::Not(_) => Ok(true),
    }
}

fn row_group_may_match_like<T: ChunkReader + 'static>(
    builder: &ParquetRecordBatchReaderBuilder<T>,
    row_group_index: usize,
    column_name: &str,
    pattern: &str,
    negated: bool,
    escape: Option<char>,
) -> Result<bool> {
    let Some((prefix, upper_bound)) = like_prefix_pruning_range(pattern, escape) else {
        return Ok(true);
    };
    if negated {
        return Ok(true);
    }

    let Some(column_index) = builder
        .schema()
        .fields()
        .iter()
        .position(|field| field.name() == column_name)
    else {
        return Ok(true);
    };
    if !matches!(
        builder.schema().field(column_index).data_type(),
        DataType::Utf8
    ) {
        return Ok(true);
    }

    let Some(row_group) = builder.metadata().row_groups().get(row_group_index) else {
        return Ok(true);
    };
    let Some(column) = row_group.columns().get(column_index) else {
        return Ok(true);
    };
    let Some(statistics) = column.statistics() else {
        return Ok(true);
    };
    if statistics.is_min_max_deprecated()
        || !statistics.min_is_exact()
        || !statistics.max_is_exact()
    {
        return Ok(true);
    }
    let Statistics::ByteArray(statistics) = statistics else {
        return Ok(true);
    };

    Ok(prefix_range_may_match(
        statistics.min_opt().map(|value| value.as_ref()),
        statistics.max_opt().map(|value| value.as_ref()),
        prefix.as_bytes(),
        upper_bound.as_deref(),
    ))
}

fn row_group_may_match_comparison<T: ChunkReader + 'static>(
    builder: &ParquetRecordBatchReaderBuilder<T>,
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

fn like_prefix_pruning_range(
    pattern: &str,
    escape: Option<char>,
) -> Option<(&str, Option<Vec<u8>>)> {
    if escape.is_some() || pattern.contains('_') || pattern.starts_with('%') {
        return None;
    }
    let prefix = pattern.strip_suffix('%')?;
    if prefix.is_empty() || prefix.contains('%') || !prefix.is_ascii() {
        return None;
    }
    Some((prefix, ascii_prefix_upper_bound(prefix)))
}

fn ascii_prefix_upper_bound(prefix: &str) -> Option<Vec<u8>> {
    let mut bytes = prefix.as_bytes().to_vec();
    for index in (0..bytes.len()).rev() {
        if bytes[index] != u8::MAX {
            bytes[index] += 1;
            bytes.truncate(index + 1);
            return Some(bytes);
        }
    }
    None
}

fn prefix_range_may_match(
    min: Option<&[u8]>,
    max: Option<&[u8]>,
    lower_bound: &[u8],
    upper_bound: Option<&[u8]>,
) -> bool {
    max.is_none_or(|max| max >= lower_bound)
        && upper_bound.is_none_or(|upper_bound| min.is_none_or(|min| min < upper_bound))
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
