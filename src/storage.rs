use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;
use std::time::SystemTime;

use arrow::array::{Array, BooleanArray, BooleanBuilder, Int64Array};
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
use parquet::column::page::Page;
use parquet::column::reader::{ColumnReader, ColumnReaderImpl};
use parquet::data_type::{ByteArray, ByteArrayType, FixedLenByteArray, Int32Type, Int64Type};
use parquet::errors::{ParquetError, Result as ParquetResult};
use parquet::file::metadata::PageIndexPolicy;
use parquet::file::reader::{ChunkReader, FileReader as ParquetFileReader, Length};
use parquet::file::serialized_reader::SerializedFileReader;
use parquet::file::statistics::Statistics;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::hash::{Hash, Hasher};

use crate::error::{DodamError, Result};
use crate::execution::{
    ComparisonExpr, ComparisonOp, Expr, FilterExpr, Projection, evaluate_filter_mask,
};
use crate::vector::RawColumnView;

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
}

pub(crate) type DirectI64I32I32ScanMetrics = DirectPrimitiveColumnScanMetrics;

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

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct DirectColumnScanMetrics {
    pub row_groups: usize,
    pub batches: usize,
    pub rows: usize,
    pub read_nanos: u64,
    pub consume_nanos: u64,
}

impl DirectColumnScanMetrics {
    fn add_read_nanos(&mut self, nanos: u64) {
        self.read_nanos = self.read_nanos.saturating_add(nanos);
    }

    fn add_consume_nanos(&mut self, nanos: u64) {
        self.consume_nanos = self.consume_nanos.saturating_add(nanos);
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
                Some(&mut predicate_def_levels),
                None,
                &mut predicate_values,
            )?;
            if records == 0 {
                metrics.add_read_nanos(elapsed_nanos(read_started));
                break;
            }
            let (sum_records, sum_value_count, _) = sum_reader.read_records(
                records,
                Some(&mut sum_def_levels),
                None,
                &mut sum_values,
            )?;
            metrics.add_read_nanos(elapsed_nanos(read_started));
            if sum_records != records
                || predicate_value_count != records
                || sum_value_count != records
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
        return Ok(());
    }
    Ok(())
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
    let before = output.len();
    let mut temp = Vec::with_capacity(values);
    decode_rle_i32_values(data, bit_width, values, &mut temp)?;
    output.extend(temp.into_iter().map(|value| value as i16));
    if output.len() != before + values {
        return Ok(());
    }
    Ok(())
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
    let mut decoder = SimpleRleBitpackedDecoder::new(data, bit_width);
    output.reserve(values);
    for _ in 0..values {
        let Some(value) = decoder.next_value()? else {
            break;
        };
        output.push(value as i32);
    }
    Ok(())
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
                Some(&mut predicate_def_levels),
                None,
                &mut predicate_values,
            )?;
            if records == 0 {
                metrics.add_read_nanos(elapsed_nanos(read_started));
                break;
            }
            let (sum_records, sum_value_count, _) = sum_reader.read_records(
                records,
                Some(&mut sum_def_levels),
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
    consume: F,
) -> Result<Option<DirectPrimitiveColumnScanMetrics>>
where
    F: for<'a> FnMut(&[RawColumnView<'a>]) -> Result<()>,
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
fn scan_parquet_i32_i64_decimal_i32_selected_reader<R, F>(
    reader: SerializedFileReader<R>,
    batch_size: usize,
    row_groups: &[usize],
    columns: [&str; 4],
    decimal_precision: u8,
    decimal_scale: i8,
    decimal_min: Option<i64>,
    decimal_max: Option<i64>,
    date_min: Option<i32>,
    date_max: Option<i32>,
    mut consume: F,
) -> Result<Option<DirectPrimitiveColumnScanMetrics>>
where
    R: ChunkReader + 'static,
    F: for<'a> FnMut(&[RawColumnView<'a>]) -> Result<()>,
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
    for &column_index in &[key_column, sum_column, decimal_column, date_column] {
        if schema.column(column_index).max_def_level() != 0 {
            return Ok(None);
        }
    }
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
        let mut selected_keys = Vec::<i32>::with_capacity(batch_size);
        let mut selected_sums = Vec::<i64>::with_capacity(batch_size);
        let mut selected_decimals = Vec::<i64>::with_capacity(batch_size);
        let mut selected_dates = Vec::<i32>::with_capacity(batch_size);
        let mut selected_runs = Vec::<(usize, usize)>::new();
        loop {
            decimal_values.clear();
            date_values.clear();
            selected_keys.clear();
            selected_sums.clear();
            selected_decimals.clear();
            selected_dates.clear();
            selected_runs.clear();
            let read_started = Instant::now();
            let decimal_started = Instant::now();
            let (records, value_count, level_count) =
                decimal_reader.read_records(batch_size, None, None, &mut decimal_values)?;
            metrics.add_column_read_nanos(2, elapsed_nanos(decimal_started));
            if records == 0 {
                metrics.add_read_nanos(elapsed_nanos(read_started));
                break;
            }
            if value_count != records || !direct_def_levels_match(level_count, records, true) {
                metrics.add_read_nanos(elapsed_nanos(read_started));
                return Ok(None);
            }
            let date_started = Instant::now();
            let (date_records, date_value_count, date_level_count) =
                date_reader.read_records(records, None, None, &mut date_values)?;
            metrics.add_column_read_nanos(3, elapsed_nanos(date_started));
            if date_records != records
                || date_value_count != records
                || !direct_def_levels_match(date_level_count, records, true)
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
                    &mut selected_keys,
                )? {
                    metrics.add_read_nanos(elapsed_nanos(read_started));
                    return Ok(None);
                }
            } else {
                let (key_records, key_value_count, key_level_count) =
                    key_reader.read_records(records, None, None, &mut selected_keys)?;
                if key_records != records
                    || key_value_count != records
                    || !direct_def_levels_match(key_level_count, records, true)
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
                    &mut selected_sums,
                )? {
                    metrics.add_read_nanos(elapsed_nanos(read_started));
                    return Ok(None);
                }
                metrics.selected_payload_batches += 1;
            } else {
                let (sum_records, sum_value_count, sum_level_count) =
                    sum_reader.read_records(records, None, None, &mut selected_sums)?;
                if sum_records != records
                    || sum_value_count != records
                    || !direct_def_levels_match(sum_level_count, records, true)
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
            let decimal_view = if use_selected_payload {
                selected_decimals.as_slice()
            } else {
                decimal_values.as_slice()
            };
            let date_view = if use_selected_payload {
                selected_dates.as_slice()
            } else {
                date_values.as_slice()
            };
            let views = [
                RawColumnView::I32(&selected_keys),
                RawColumnView::I64(&selected_sums),
                RawColumnView::Decimal128I64 {
                    values: decimal_view,
                    precision: decimal_precision,
                    scale: decimal_scale,
                },
                RawColumnView::Date32(date_view),
            ];
            consume(&views)?;
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
    output: &mut Vec<i32>,
) -> Result<bool> {
    let mut cursor = 0usize;
    for &(start, len) in runs {
        if start > cursor && reader.skip_records(start - cursor)? != start - cursor {
            return Ok(false);
        }
        let (read_records, value_count, level_count) =
            reader.read_records(len, None, None, output)?;
        if read_records != len || value_count != len || level_count != 0 {
            return Ok(false);
        }
        cursor = start + len;
    }
    if records > cursor && reader.skip_records(records - cursor)? != records - cursor {
        return Ok(false);
    }
    Ok(true)
}

fn read_i64_selected_runs(
    reader: &mut ColumnReaderImpl<Int64Type>,
    records: usize,
    runs: &[(usize, usize)],
    output: &mut Vec<i64>,
) -> Result<bool> {
    let mut cursor = 0usize;
    for &(start, len) in runs {
        if start > cursor && reader.skip_records(start - cursor)? != start - cursor {
            return Ok(false);
        }
        let (read_records, value_count, level_count) =
            reader.read_records(len, None, None, output)?;
        if read_records != len || value_count != len || level_count != 0 {
            return Ok(false);
        }
        cursor = start + len;
    }
    if records > cursor && reader.skip_records(records - cursor)? != records - cursor {
        return Ok(false);
    }
    Ok(true)
}

fn direct_selection_payload_gate(
    records: usize,
    selected_rows: usize,
    selected_runs: usize,
) -> bool {
    if records == 0 || selected_rows == 0 {
        log_direct_selection_gate(
            records,
            selected_rows,
            selected_runs,
            false,
            "empty-selection",
        );
        return false;
    }
    let max_ratio = std::env::var("DODAM_DIRECT_SELECTION_MAX_RATIO")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(0.20);
    let min_run_len = std::env::var("DODAM_DIRECT_SELECTION_MIN_RUN_LEN")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(32);
    let selected_ratio = selected_rows as f64 / records as f64;
    let average_run_len = selected_rows / selected_runs.max(1);
    let accepted = selected_ratio <= max_ratio && average_run_len >= min_run_len;
    let reason = if accepted {
        "accepted"
    } else if selected_ratio > max_ratio {
        "selected-ratio"
    } else {
        "fragmented-runs"
    };
    log_direct_selection_gate(records, selected_rows, selected_runs, accepted, reason);
    accepted
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
        indices.push(
            batch
                .schema()
                .fields()
                .iter()
                .position(|field| field.name() == column)
                .ok_or_else(|| DodamError::UnknownColumn(column.clone()))?,
        );
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
        Expr::Boolean(_)
        | Expr::ColumnComparison { .. }
        | Expr::InList { .. }
        | Expr::Like { .. }
        | Expr::IsNull { .. }
        | Expr::Not(_) => Ok(true),
    }
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
