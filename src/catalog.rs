use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use arrow::datatypes::SchemaRef;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde::{Deserialize, Serialize};

use crate::error::{DodamError, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileFragment {
    pub location: StorageLocation,
    pub format: StorageFormat,
    pub statistics: Option<FileFragmentStatistics>,
    pub partition_values: BTreeMap<String, String>,
}

impl FileFragment {
    pub fn new(location: StorageLocation, format: StorageFormat) -> Self {
        Self {
            location,
            format,
            statistics: None,
            partition_values: BTreeMap::new(),
        }
    }

    pub fn local_parquet(path: impl Into<PathBuf>) -> Self {
        Self {
            location: StorageLocation::LocalPath(path.into()),
            format: StorageFormat::Parquet,
            statistics: None,
            partition_values: BTreeMap::new(),
        }
    }

    pub fn with_partition_values(mut self, partition_values: BTreeMap<String, String>) -> Self {
        self.partition_values = partition_values;
        self
    }

    pub fn with_statistics(mut self, statistics: FileFragmentStatistics) -> Self {
        self.statistics = Some(statistics);
        self
    }

    pub fn parquet_local_path(&self) -> Result<&Path> {
        self.require_format(StorageFormat::Parquet)?;
        match &self.location {
            StorageLocation::LocalPath(path) => Ok(path),
            StorageLocation::ObjectUri(uri) => {
                Err(DodamError::UnsupportedStorageLocation(uri.clone()))
            }
        }
    }

    pub fn require_format(&self, expected: StorageFormat) -> Result<()> {
        if self.format == expected {
            return Ok(());
        }
        Err(DodamError::UnsupportedStorageFormat(format!(
            "{:?}",
            self.format
        )))
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileFragmentStatistics {
    pub rows: usize,
    pub row_groups: usize,
    pub compressed_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum StorageLocation {
    LocalPath(PathBuf),
    ObjectUri(String),
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum StorageFormat {
    #[default]
    Parquet,
    Csv,
    Json,
    ArrowIpc,
}

#[derive(Debug, Clone)]
pub struct TableScanSource {
    pub fragments: Vec<FileFragment>,
    pub schema: Option<SchemaRef>,
    pub format: StorageFormat,
    pub statistics: TableStatistics,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TableStatistics {
    pub fragments: usize,
    pub rows: usize,
    pub row_groups: usize,
    pub compressed_bytes: u64,
}

impl TableStatistics {
    pub fn from_fragments(fragments: &[FileFragment]) -> Self {
        fragments
            .iter()
            .filter_map(|fragment| fragment.statistics)
            .fold(Self::default(), |mut table, fragment| {
                table.fragments += 1;
                table.rows = table.rows.saturating_add(fragment.rows);
                table.row_groups = table.row_groups.saturating_add(fragment.row_groups);
                table.compressed_bytes = table
                    .compressed_bytes
                    .saturating_add(fragment.compressed_bytes);
                table
            })
    }
}

pub trait TableProvider: Send + Sync {
    fn scan_source(&self) -> Result<TableScanSource>;
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CatalogTableEntry {
    pub name: String,
    pub location: String,
    pub format: CatalogStorageFormat,
    #[serde(default)]
    pub metadata: Option<CatalogTableMetadata>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CatalogTableMetadata {
    pub schema: Vec<CatalogSchemaField>,
    pub statistics: TableStatistics,
    #[serde(default)]
    pub partition_columns: Vec<String>,
    #[serde(default)]
    pub fragments: Vec<CatalogFileFragment>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CatalogSchemaField {
    pub name: String,
    pub data_type: String,
    pub nullable: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CatalogFileFragment {
    pub location: String,
    pub format: StorageFormat,
    pub statistics: FileFragmentStatistics,
    #[serde(default)]
    pub partition_values: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum CatalogStorageFormat {
    Parquet,
}

impl From<CatalogStorageFormat> for StorageFormat {
    fn from(format: CatalogStorageFormat) -> Self {
        match format {
            CatalogStorageFormat::Parquet => StorageFormat::Parquet,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct CatalogFile {
    #[serde(default)]
    pub tables: BTreeMap<String, CatalogTableEntry>,
}

#[derive(Debug, Clone)]
pub struct PersistentCatalog {
    root: PathBuf,
}

impl PersistentCatalog {
    pub const DIRECTORY: &'static str = ".dodam";
    pub const FILE_NAME: &'static str = "catalog.json";

    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn catalog_dir(&self) -> PathBuf {
        self.root.join(Self::DIRECTORY)
    }

    pub fn catalog_path(&self) -> PathBuf {
        self.catalog_dir().join(Self::FILE_NAME)
    }

    pub fn load(&self) -> Result<CatalogFile> {
        let path = self.catalog_path();
        if !path.exists() {
            return Ok(CatalogFile::default());
        }
        let contents = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&contents)?)
    }

    pub fn save(&self, catalog: &CatalogFile) -> Result<()> {
        std::fs::create_dir_all(self.catalog_dir())?;
        let contents = serde_json::to_string_pretty(catalog)?;
        std::fs::write(self.catalog_path(), format!("{contents}\n"))?;
        Ok(())
    }

    pub fn register_local_parquet(
        &self,
        name: impl Into<String>,
        location: impl AsRef<Path>,
    ) -> Result<CatalogTableEntry> {
        let name = normalize_table_name(&name.into())?;
        let location = location.as_ref().to_path_buf();
        if !location.exists() {
            return Err(DodamError::MissingPath(location));
        }
        let mut catalog = self.load()?;
        let metadata = inspect_local_parquet_table(&location)?;
        let entry = CatalogTableEntry {
            name: name.clone(),
            location: location.to_string_lossy().to_string(),
            format: CatalogStorageFormat::Parquet,
            metadata: Some(metadata),
        };
        catalog.tables.insert(name, entry.clone());
        self.save(&catalog)?;
        Ok(entry)
    }

    pub fn refresh_table(&self, name: &str) -> Result<CatalogTableEntry> {
        let name = normalize_table_name(name)?;
        let mut catalog = self.load()?;
        let entry =
            catalog.tables.get(&name).cloned().ok_or_else(|| {
                DodamError::UnsupportedSql(format!("unknown catalog table: {name}"))
            })?;
        let location = PathBuf::from(&entry.location);
        if !location.exists() {
            return Err(DodamError::MissingPath(location));
        }
        let metadata = match entry.format {
            CatalogStorageFormat::Parquet => inspect_local_parquet_table(&location)?,
        };
        let refreshed = CatalogTableEntry {
            metadata: Some(metadata),
            ..entry
        };
        catalog.tables.insert(name, refreshed.clone());
        self.save(&catalog)?;
        Ok(refreshed)
    }

    pub fn table(&self, name: &str) -> Result<Option<CatalogTableEntry>> {
        let name = normalize_table_name(name)?;
        Ok(self.load()?.tables.get(&name).cloned())
    }

    pub fn table_scan_source(&self, name: &str) -> Result<Option<TableScanSource>> {
        let Some(entry) = self.table(name)? else {
            return Ok(None);
        };
        let Some(metadata) = entry.metadata else {
            return Ok(None);
        };
        if metadata.fragments.is_empty() {
            return Ok(None);
        }
        Ok(Some(TableScanSource {
            fragments: metadata
                .fragments
                .into_iter()
                .map(|fragment| {
                    FileFragment::new(
                        StorageLocation::LocalPath(PathBuf::from(fragment.location)),
                        fragment.format,
                    )
                    .with_statistics(fragment.statistics)
                    .with_partition_values(fragment.partition_values)
                })
                .collect(),
            schema: None,
            format: StorageFormat::from(entry.format),
            statistics: metadata.statistics,
        }))
    }

    pub fn tables(&self) -> Result<Vec<CatalogTableEntry>> {
        Ok(self.load()?.tables.into_values().collect())
    }
}

fn normalize_table_name(name: &str) -> Result<String> {
    let name = name.trim();
    if name.is_empty() {
        return Err(DodamError::UnsupportedSql(
            "table name must not be empty".to_string(),
        ));
    }
    if !name
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
    {
        return Err(DodamError::UnsupportedSql(format!(
            "unsupported table name: {name}"
        )));
    }
    Ok(name.to_ascii_lowercase())
}

#[derive(Debug, Clone)]
pub struct LocalParquetTable {
    root: PathBuf,
}

impl LocalParquetTable {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    fn discover_parquet_files(path: &Path, files: &mut Vec<FileFragment>) -> Result<()> {
        Self::discover_parquet_files_from_root(path, path, files)
    }

    fn discover_parquet_files_from_root(
        root: &Path,
        path: &Path,
        files: &mut Vec<FileFragment>,
    ) -> Result<()> {
        if path.is_file() {
            if is_parquet_file(path) {
                files.push(
                    FileFragment::local_parquet(path)
                        .with_partition_values(partition_values_for_path(root, path)),
                );
                return Ok(());
            }

            return Err(DodamError::UnsupportedTablePath(path.to_path_buf()));
        }

        if path.is_dir() {
            for entry in std::fs::read_dir(path)? {
                let entry = entry?;
                let child = entry.path();
                if child.is_dir() {
                    Self::discover_parquet_files_from_root(root, &child, files)?;
                } else if is_parquet_file(&child) {
                    files.push(
                        FileFragment::local_parquet(&child)
                            .with_partition_values(partition_values_for_path(root, &child)),
                    );
                }
            }
            files.sort_by(|left, right| left.location.cmp(&right.location));
            return Ok(());
        }

        Err(DodamError::MissingPath(path.to_path_buf()))
    }
}

fn inspect_local_parquet_table(path: &Path) -> Result<CatalogTableMetadata> {
    let mut fragments = Vec::new();
    LocalParquetTable::discover_parquet_files(path, &mut fragments)?;
    let mut schema: Option<SchemaRef> = None;
    let mut statistics = TableStatistics::default();
    let mut partition_columns = BTreeMap::new();
    let mut catalog_fragments = Vec::new();
    for fragment in fragments {
        for column in fragment.partition_values.keys() {
            partition_columns.insert(column.clone(), ());
        }
        let path = fragment.parquet_local_path()?;
        let file = File::open(path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let fragment_schema = builder.schema().clone();
        match &schema {
            Some(existing) if existing.as_ref() != fragment_schema.as_ref() => {
                return Err(DodamError::UnsupportedSql(
                    "table fragments must have identical schemas".to_string(),
                ));
            }
            Some(_) => {}
            None => schema = Some(fragment_schema),
        }
        let row_groups = builder.metadata().num_row_groups();
        let rows = builder
            .metadata()
            .row_groups()
            .iter()
            .map(|row_group| usize::try_from(row_group.num_rows()).unwrap_or(usize::MAX))
            .fold(0_usize, usize::saturating_add);
        let compressed_bytes = compressed_bytes_for_all_columns(&builder);
        let fragment_statistics = FileFragmentStatistics {
            rows,
            row_groups,
            compressed_bytes,
        };
        statistics.fragments = statistics.fragments.saturating_add(1);
        statistics.row_groups = statistics.row_groups.saturating_add(row_groups);
        statistics.rows = statistics.rows.saturating_add(rows);
        statistics.compressed_bytes = statistics.compressed_bytes.saturating_add(compressed_bytes);
        catalog_fragments.push(CatalogFileFragment {
            location: path.to_string_lossy().to_string(),
            format: fragment.format,
            statistics: fragment_statistics,
            partition_values: fragment.partition_values,
        });
    }
    let schema = schema.ok_or_else(|| DodamError::UnsupportedTablePath(path.to_path_buf()))?;
    Ok(CatalogTableMetadata {
        schema: schema
            .fields()
            .iter()
            .map(|field| CatalogSchemaField {
                name: field.name().clone(),
                data_type: format!("{:?}", field.data_type()),
                nullable: field.is_nullable(),
            })
            .collect(),
        statistics,
        partition_columns: partition_columns.into_keys().collect(),
        fragments: catalog_fragments,
    })
}

fn compressed_bytes_for_all_columns(builder: &ParquetRecordBatchReaderBuilder<File>) -> u64 {
    builder
        .metadata()
        .row_groups()
        .iter()
        .flat_map(|row_group| row_group.columns())
        .map(|column| u64::try_from(column.compressed_size()).unwrap_or_default())
        .fold(0_u64, u64::saturating_add)
}

fn partition_values_for_path(root: &Path, path: &Path) -> BTreeMap<String, String> {
    let partition_root = if root.is_file() {
        root.parent().unwrap_or(root)
    } else {
        root
    };
    let Ok(relative) = path.strip_prefix(partition_root) else {
        return BTreeMap::new();
    };
    relative
        .parent()
        .into_iter()
        .flat_map(|parent| parent.components())
        .filter_map(|component| {
            let text = component.as_os_str().to_str()?;
            let (key, value) = text.split_once('=')?;
            (!key.is_empty() && !value.is_empty()).then(|| (key.to_string(), value.to_string()))
        })
        .collect()
}

impl TableProvider for LocalParquetTable {
    fn scan_source(&self) -> Result<TableScanSource> {
        let mut files = Vec::new();
        Self::discover_parquet_files(&self.root, &mut files)?;
        Ok(TableScanSource {
            fragments: files,
            schema: None,
            format: StorageFormat::Parquet,
            statistics: TableStatistics::default(),
        })
    }
}

fn is_parquet_file(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("parquet"))
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::sync::Arc;

    use arrow::array::{Int32Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;

    use crate::catalog::TableProvider;

    use super::{LocalParquetTable, PersistentCatalog};

    #[test]
    fn persistent_catalog_registers_tables_under_dodam_directory() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let data_dir = tempdir.path().join("orders");
        let partition_dir = data_dir.join("dt=2026-07-01").join("country=kr");
        let refreshed_partition_dir = data_dir.join("dt=2026-07-02").join("country=kr");
        std::fs::create_dir_all(&partition_dir).expect("data dir");
        write_catalog_test_parquet(&partition_dir.join("part-000.parquet"));
        let catalog = PersistentCatalog::new(tempdir.path());

        let entry = catalog
            .register_local_parquet("Orders", &data_dir)
            .expect("register table");

        assert_eq!(entry.name, "orders");
        let metadata = entry.metadata.expect("metadata");
        assert_eq!(metadata.statistics.fragments, 1);
        assert_eq!(metadata.statistics.rows, 3);
        assert_eq!(metadata.partition_columns, vec!["country", "dt"]);
        assert_eq!(metadata.schema[0].name, "id");
        assert_eq!(metadata.fragments.len(), 1);
        assert_eq!(
            metadata.fragments[0].partition_values.get("country"),
            Some(&"kr".to_string())
        );
        assert_eq!(metadata.fragments[0].statistics.rows, 3);
        assert!(catalog.catalog_path().exists());
        assert_eq!(
            catalog
                .table("orders")
                .expect("lookup")
                .expect("table")
                .location,
            data_dir.to_string_lossy()
        );
        assert_eq!(catalog.tables().expect("tables").len(), 1);

        std::fs::create_dir_all(&refreshed_partition_dir).expect("refresh dir");
        write_catalog_test_parquet(&refreshed_partition_dir.join("part-000.parquet"));
        let refreshed = catalog.refresh_table("orders").expect("refresh table");
        let metadata = refreshed.metadata.expect("refreshed metadata");
        assert_eq!(metadata.statistics.fragments, 2);
        assert_eq!(metadata.fragments.len(), 2);
    }

    #[test]
    fn local_parquet_table_discovers_hive_partition_values() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let table_dir = tempdir.path().join("events");
        let partition_dir = table_dir.join("dt=2026-07-01").join("country=kr");
        std::fs::create_dir_all(&partition_dir).expect("partition dir");
        write_catalog_test_parquet(&partition_dir.join("part-000.parquet"));

        let source = LocalParquetTable::new(&table_dir)
            .scan_source()
            .expect("scan source");

        assert_eq!(source.fragments.len(), 1);
        assert_eq!(
            source.fragments[0].partition_values.get("dt"),
            Some(&"2026-07-01".to_string())
        );
        assert_eq!(
            source.fragments[0].partition_values.get("country"),
            Some(&"kr".to_string())
        );
    }

    fn write_catalog_test_parquet(path: &std::path::Path) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("payload", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec!["a", "b", "c"])),
            ],
        )
        .expect("record batch");
        let file = File::create(path).expect("create parquet");
        let mut writer = ArrowWriter::try_new(file, schema, None).expect("writer");
        writer.write(&batch).expect("write batch");
        writer.close().expect("close writer");
    }
}
