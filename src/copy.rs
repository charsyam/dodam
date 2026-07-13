use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow::array::{
    Array, BooleanArray, Float64Array, Int32Array, Int64Array, StringArray, UInt32Array,
    UInt64Array,
};
use arrow::datatypes::{DataType, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, Encoding};
use parquet::data_type::{Int32Type, Int64Type};
use parquet::file::properties::{EnabledStatistics, WriterProperties, WriterPropertiesBuilder};
use parquet::file::writer::SerializedFileWriter;
use parquet::schema::parser::parse_message_type;
use parquet::schema::types::{ColumnPath, TypePtr};
use sqlparser::ast::{CopyOption, CopySource, CopyTarget, Ident, Statement};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser as SqlParser;

use crate::cost::{PrimitiveParquetSinkCostInput, choose_primitive_parquet_sink};
use crate::error::{DodamError, Result};
use crate::execution::{PrimitiveBatch, PrimitiveColumnValues, RecordBatchSink, ScanPlanMetrics};
use crate::sql::{QueryOutput, SqlResultSink};

#[derive(Debug, Clone)]
pub struct CopyToSelect {
    pub sql: String,
    pub path: PathBuf,
    pub header: bool,
    pub format: CopyFormat,
    pub parquet_options: ParquetCopyOptions,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CopyFormat {
    Csv,
    Parquet,
}

#[derive(Debug, Clone, Copy)]
pub struct ParquetCopyOptions {
    pub compression: Compression,
    pub dictionary_enabled: bool,
    pub max_row_group_rows: Option<usize>,
    pub write_batch_size: usize,
    pub data_page_row_count_limit: usize,
}

impl Default for ParquetCopyOptions {
    fn default() -> Self {
        Self {
            compression: Compression::SNAPPY,
            dictionary_enabled: false,
            max_row_group_rows: Some(256 * 1024),
            write_batch_size: 32 * 1024,
            data_page_row_count_limit: 16 * 1024,
        }
    }
}

pub fn parse_copy_to_select(sql: &str) -> Result<Option<CopyToSelect>> {
    let dialect = GenericDialect {};
    let mut parquet_options = ParquetCopyOptions::default();
    let sanitized_sql = sanitize_copy_options(sql, &mut parquet_options)?;
    let statements = SqlParser::parse_sql(&dialect, &sanitized_sql)
        .map_err(|error| DodamError::UnsupportedSql(error.to_string()))?;
    let [statement] = statements.as_slice() else {
        return Err(DodamError::UnsupportedSql(
            "expected exactly one statement".to_string(),
        ));
    };

    let Statement::Copy {
        source,
        to,
        target,
        options,
        legacy_options,
        values,
    } = statement
    else {
        return Ok(None);
    };
    if !to {
        return Err(DodamError::UnsupportedSql(
            "COPY FROM is not supported".to_string(),
        ));
    }
    if !legacy_options.is_empty() || !values.is_empty() {
        return Err(DodamError::UnsupportedSql(
            "COPY legacy options and inline values are not supported".to_string(),
        ));
    }

    let CopySource::Query(query) = source else {
        return Err(DodamError::UnsupportedSql(
            "COPY TO currently supports only COPY (SELECT ...)".to_string(),
        ));
    };
    let CopyTarget::File { filename } = target else {
        return Err(DodamError::UnsupportedSql(
            "COPY TO currently supports only file targets".to_string(),
        ));
    };

    let mut format = CopyFormat::Csv;
    let mut header = false;
    for option in options {
        match option {
            CopyOption::Format(ident) if ident_eq(ident, "csv") => format = CopyFormat::Csv,
            CopyOption::Format(ident) if ident_eq(ident, "parquet") => format = CopyFormat::Parquet,
            CopyOption::Header(value) => header = *value,
            CopyOption::Format(ident) => {
                return Err(DodamError::UnsupportedSql(format!(
                    "COPY FORMAT {ident} is not supported"
                )));
            }
            option => {
                return Err(DodamError::UnsupportedSql(format!(
                    "COPY option {option} is not supported"
                )));
            }
        }
    }
    if format == CopyFormat::Parquet && header {
        return Err(DodamError::UnsupportedSql(
            "COPY HEADER is only supported for FORMAT CSV".to_string(),
        ));
    }

    Ok(Some(CopyToSelect {
        sql: query.to_string(),
        path: PathBuf::from(filename),
        header,
        format,
        parquet_options,
    }))
}

fn sanitize_copy_options(sql: &str, parquet_options: &mut ParquetCopyOptions) -> Result<String> {
    if !sql.trim_start().to_ascii_uppercase().starts_with("COPY ") {
        return Ok(sql.to_string());
    }

    let mut end = sql.trim_end().len();
    let suffix = if sql[..end].ends_with(';') {
        end -= 1;
        ";"
    } else {
        ""
    };
    let body = &sql[..end];
    if !body.trim_end().ends_with(')') {
        return Ok(sql.to_string());
    }

    let close = body.trim_end().len() - 1;
    let Some(open) = matching_open_paren(body, close) else {
        return Ok(sql.to_string());
    };
    let options = &body[open + 1..close];
    if !copy_options_need_sanitizing(options) {
        return Ok(sql.to_string());
    }

    let mut kept = Vec::new();
    for option in split_copy_options(options) {
        if !parse_parquet_copy_option(&option, parquet_options)? {
            kept.push(option.trim().to_string());
        }
    }

    let prefix = body[..open].trim_end();
    if kept.is_empty() {
        Ok(format!("{prefix}{suffix}"))
    } else {
        Ok(format!("{prefix} ({}){suffix}", kept.join(", ")))
    }
}

fn copy_options_need_sanitizing(options: &str) -> bool {
    let upper = options.to_ascii_uppercase();
    upper.contains("COMPRESSION")
        || upper.contains("DICTIONARY")
        || upper.contains("ROW_GROUP")
        || upper.contains("WRITE_BATCH_SIZE")
        || upper.contains("DATA_PAGE_ROW_COUNT_LIMIT")
        || upper.contains("PAGE_ROW_COUNT_LIMIT")
}

fn matching_open_paren(sql: &str, close: usize) -> Option<usize> {
    let mut stack = Vec::new();
    let mut in_quote = false;
    let bytes = sql.as_bytes();
    let mut index = 0;
    while index <= close {
        match bytes[index] {
            b'\'' => {
                if in_quote && bytes.get(index + 1) == Some(&b'\'') {
                    index += 1;
                } else {
                    in_quote = !in_quote;
                }
            }
            b'(' if !in_quote => stack.push(index),
            b')' if !in_quote => {
                let open = stack.pop()?;
                if index == close {
                    return Some(open);
                }
            }
            _ => {}
        }
        index += 1;
    }
    None
}

fn split_copy_options(options: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut start = 0;
    let mut in_quote = false;
    let bytes = options.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            b'\'' => {
                if in_quote && bytes.get(index + 1) == Some(&b'\'') {
                    index += 1;
                } else {
                    in_quote = !in_quote;
                }
            }
            b',' if !in_quote => {
                parts.push(options[start..index].to_string());
                start = index + 1;
            }
            _ => {}
        }
        index += 1;
    }
    parts.push(options[start..].to_string());
    parts
}

fn parse_parquet_copy_option(
    option: &str,
    parquet_options: &mut ParquetCopyOptions,
) -> Result<bool> {
    let option = option.trim();
    if option.is_empty() {
        return Ok(true);
    }
    let (name, value) = split_option_name_value(option);
    match name.to_ascii_uppercase().as_str() {
        "COMPRESSION" => {
            parquet_options.compression = parse_parquet_compression(value)?;
            Ok(true)
        }
        "DICTIONARY" | "DICTIONARY_ENABLED" => {
            parquet_options.dictionary_enabled = parse_copy_bool(value)?;
            Ok(true)
        }
        "ROW_GROUP_SIZE" | "ROW_GROUP_ROWS" | "MAX_ROW_GROUP_ROW_COUNT" => {
            parquet_options.max_row_group_rows = Some(parse_copy_usize(value, "ROW_GROUP_SIZE")?);
            Ok(true)
        }
        "WRITE_BATCH_SIZE" => {
            parquet_options.write_batch_size = parse_copy_usize(value, "WRITE_BATCH_SIZE")?;
            Ok(true)
        }
        "DATA_PAGE_ROW_COUNT_LIMIT" | "PAGE_ROW_COUNT_LIMIT" => {
            parquet_options.data_page_row_count_limit =
                parse_copy_usize(value, "DATA_PAGE_ROW_COUNT_LIMIT")?;
            Ok(true)
        }
        _ => Ok(false),
    }
}

fn split_option_name_value(option: &str) -> (&str, &str) {
    if let Some((name, value)) = option.split_once('=') {
        return (name.trim(), value.trim());
    }
    let mut split = option.splitn(2, char::is_whitespace);
    let name = split.next().unwrap_or_default();
    let value = split.next().unwrap_or_default().trim();
    (name, value)
}

fn clean_copy_option_value(value: &str) -> &str {
    let value = value.trim();
    value
        .strip_prefix('\'')
        .and_then(|value| value.strip_suffix('\''))
        .unwrap_or(value)
        .trim()
}

fn parse_parquet_compression(value: &str) -> Result<Compression> {
    match clean_copy_option_value(value).to_ascii_uppercase().as_str() {
        "SNAPPY" => Ok(Compression::SNAPPY),
        "ZSTD" => Ok(Compression::ZSTD(Default::default())),
        "UNCOMPRESSED" | "NONE" => Ok(Compression::UNCOMPRESSED),
        other => Err(DodamError::UnsupportedSql(format!(
            "COPY PARQUET COMPRESSION {other} is not supported"
        ))),
    }
}

fn parse_copy_bool(value: &str) -> Result<bool> {
    match clean_copy_option_value(value).to_ascii_uppercase().as_str() {
        "TRUE" | "ON" | "1" => Ok(true),
        "FALSE" | "OFF" | "0" => Ok(false),
        other => Err(DodamError::UnsupportedSql(format!(
            "expected boolean COPY option value, found {other}"
        ))),
    }
}

fn parse_copy_usize(value: &str, option_name: &str) -> Result<usize> {
    let value = clean_copy_option_value(value)
        .parse::<usize>()
        .map_err(|_| {
            DodamError::UnsupportedSql(format!("{option_name} expects a positive integer"))
        })?;
    if value == 0 {
        return Err(DodamError::UnsupportedSql(format!(
            "{option_name} expects a positive integer"
        )));
    }
    Ok(value)
}

fn ident_eq(ident: &Ident, expected: &str) -> bool {
    ident.value.eq_ignore_ascii_case(expected)
}

pub struct CsvFileQuerySink {
    writer: BufWriter<File>,
    buffer: Vec<u8>,
    wrote_header: bool,
    header: bool,
    discard: bool,
    profile_enabled: bool,
    stats: CsvSinkStats,
}

pub enum CopyFileQuerySink {
    Csv(CsvFileQuerySink),
    Parquet(ParquetFileQuerySink),
}

impl CopyFileQuerySink {
    pub fn new(
        path: &Path,
        format: CopyFormat,
        header: bool,
        parquet_options: ParquetCopyOptions,
        copy_buffer_size: Option<usize>,
        profile_enabled: bool,
    ) -> Result<Self> {
        match format {
            CopyFormat::Csv => Ok(Self::Csv(CsvFileQuerySink::new(
                path,
                header,
                copy_buffer_size_bytes(copy_buffer_size, CSV_SINK_BUFFER_BYTES),
                profile_enabled,
            )?)),
            CopyFormat::Parquet => Ok(Self::Parquet(ParquetFileQuerySink::new(
                path,
                profile_enabled,
                parquet_options,
                copy_buffer_size,
            ))),
        }
    }

    pub fn stats(&self) -> &CsvSinkStats {
        match self {
            Self::Csv(sink) => sink.stats(),
            Self::Parquet(sink) => sink.stats(),
        }
    }

    fn write_output(&mut self, output: QueryOutput) -> Result<()> {
        match self {
            Self::Csv(sink) => sink.write_output(output),
            Self::Parquet(sink) => sink.write_output(output),
        }
    }
}

impl RecordBatchSink for CopyFileQuerySink {
    fn write_batch(&mut self, batch: &RecordBatch) -> Result<()> {
        match self {
            Self::Csv(sink) => sink.write_batch(batch),
            Self::Parquet(sink) => sink.write_batch(batch),
        }
    }

    fn write_primitive_batch(&mut self, batch: PrimitiveBatch) -> Result<bool> {
        match self {
            Self::Csv(sink) => sink.write_primitive_batch(batch),
            Self::Parquet(sink) => sink.write_primitive_batch(batch),
        }
    }

    fn supports_i32_utf8_rows(&self) -> bool {
        matches!(self, Self::Csv(sink) if sink.supports_i32_utf8_rows())
    }

    fn write_i32_utf8_rows(
        &mut self,
        left: &Int32Array,
        left_indices: &[u32],
        right_arrays: &[&StringArray],
        right_batch_indices: &[usize],
        right_row_indices: &[u32],
    ) -> Result<bool> {
        match self {
            Self::Csv(sink) => sink.write_i32_utf8_rows(
                left,
                left_indices,
                right_arrays,
                right_batch_indices,
                right_row_indices,
            ),
            Self::Parquet(_) => Ok(false),
        }
    }

    fn supports_i32_rows(&self) -> bool {
        matches!(self, Self::Csv(sink) if sink.supports_i32_rows())
    }

    fn write_i32_rows(&mut self, array: &Int32Array, indices: &[u32]) -> Result<bool> {
        match self {
            Self::Csv(sink) => sink.write_i32_rows(array, indices),
            Self::Parquet(_) => Ok(false),
        }
    }

    fn discards_output(&self) -> bool {
        match self {
            Self::Csv(sink) => sink.discards_output(),
            Self::Parquet(sink) => sink.discards_output(),
        }
    }

    fn finish(&mut self) -> Result<()> {
        match self {
            Self::Csv(sink) => sink.finish(),
            Self::Parquet(sink) => sink.finish(),
        }
    }
}

impl SqlResultSink for CopyFileQuerySink {
    fn record_batch_sink(&mut self) -> &mut dyn RecordBatchSink {
        self
    }

    fn write_output(&mut self, output: QueryOutput) -> Result<()> {
        CopyFileQuerySink::write_output(self, output)
    }
}

const CSV_SINK_BUFFER_BYTES: usize = 8 * 1024 * 1024;
const PARQUET_SINK_BUFFER_BYTES: usize = 8 * 1024 * 1024;
const PARQUET_WIDE_SINK_BUFFER_BYTES: usize = 1024 * 1024;

impl CsvFileQuerySink {
    fn new(path: &Path, header: bool, buffer_size: usize, profile_enabled: bool) -> Result<Self> {
        let file = File::create(path)?;
        let writer = BufWriter::with_capacity(buffer_size, file);
        let discard = path == Path::new("/dev/null");
        Ok(Self {
            writer,
            buffer: Vec::with_capacity(buffer_size),
            wrote_header: false,
            header,
            discard,
            profile_enabled,
            stats: CsvSinkStats::default(),
        })
    }

    fn stats(&self) -> &CsvSinkStats {
        &self.stats
    }

    fn write_output(&mut self, output: QueryOutput) -> Result<()> {
        match output {
            QueryOutput::Scan { batches } | QueryOutput::Aggregate { batches, .. } => {
                for batch in batches {
                    self.write_csv_batch(&batch)?;
                }
                Ok(())
            }
            QueryOutput::Explain { .. } => Err(DodamError::UnsupportedSql(
                "COPY TO does not support EXPLAIN output".to_string(),
            )),
        }
    }

    fn write_csv_batch(&mut self, batch: &RecordBatch) -> Result<()> {
        if self.profile_enabled {
            self.stats.batches += 1;
            self.stats.rows += batch.num_rows();
        }
        if self.discard {
            let columns_started = self.profile_enabled.then(Instant::now);
            let _columns = batch
                .columns()
                .iter()
                .map(|array| CsvColumn::try_from_array(array.as_ref(), batch.num_rows()))
                .collect::<Result<Vec<_>>>()?;
            if let Some(columns_started) = columns_started {
                self.stats.column_prepare += columns_started.elapsed();
            }
            return Ok(());
        }
        if self.header && !self.wrote_header {
            let header_started = self.profile_enabled.then(Instant::now);
            let schema = batch.schema();
            self.buffer.clear();
            for (index, field) in schema.fields().iter().enumerate() {
                if index > 0 {
                    self.buffer.push(b',');
                }
                write_csv_bytes(field.name().as_bytes(), &mut self.buffer);
            }
            self.buffer.push(b'\n');
            let bytes = self.buffer.len();
            self.writer.write_all(&self.buffer)?;
            if let Some(header_started) = header_started {
                self.stats.bytes += bytes;
                self.stats.header += header_started.elapsed();
            }
            self.wrote_header = true;
        }
        if batch.num_rows() == 0 {
            return Ok(());
        }

        let columns_started = self.profile_enabled.then(Instant::now);
        let columns = batch
            .columns()
            .iter()
            .map(|array| CsvColumn::try_from_array(array.as_ref(), batch.num_rows()))
            .collect::<Result<Vec<_>>>()?;
        if let Some(columns_started) = columns_started {
            self.stats.column_prepare += columns_started.elapsed();
        }

        let serialize_started = self.profile_enabled.then(Instant::now);
        self.buffer.clear();
        self.buffer.reserve(
            batch
                .num_rows()
                .saturating_mul(columns.len())
                .saturating_mul(16),
        );
        if !write_specialized_csv_batch(&columns, batch.num_rows(), &mut self.buffer) {
            for row in 0..batch.num_rows() {
                for (column_index, column) in columns.iter().enumerate() {
                    if column_index > 0 {
                        self.buffer.push(b',');
                    }
                    column.write_value(row, &mut self.buffer)?;
                }
                self.buffer.push(b'\n');
            }
        }
        if let Some(serialize_started) = serialize_started {
            self.stats.serialize += serialize_started.elapsed();
        }

        let write_started = self.profile_enabled.then(Instant::now);
        let bytes = self.buffer.len();
        self.writer.write_all(&self.buffer)?;
        if let Some(write_started) = write_started {
            self.stats.bytes += bytes;
            self.stats.write += write_started.elapsed();
        }
        Ok(())
    }

    fn finish_csv(&mut self) -> Result<()> {
        let flush_started = self.profile_enabled.then(Instant::now);
        self.writer.flush()?;
        if let Some(flush_started) = flush_started {
            self.stats.flush += flush_started.elapsed();
        }
        Ok(())
    }
}

pub struct ParquetFileQuerySink {
    path: PathBuf,
    writer: Option<ArrowWriter<BufWriter<File>>>,
    primitive_writer: Option<PrimitiveParquetFileWriter>,
    primitive_schema: Option<Arc<Schema>>,
    options: ParquetCopyOptions,
    buffer_size_override: Option<usize>,
    profile_enabled: bool,
    stats: CsvSinkStats,
}

impl ParquetFileQuerySink {
    fn new(
        path: &Path,
        profile_enabled: bool,
        options: ParquetCopyOptions,
        buffer_size_override: Option<usize>,
    ) -> Self {
        Self {
            path: path.to_path_buf(),
            writer: None,
            primitive_writer: None,
            primitive_schema: None,
            options,
            buffer_size_override,
            profile_enabled,
            stats: CsvSinkStats::default(),
        }
    }

    fn stats(&self) -> &CsvSinkStats {
        &self.stats
    }

    fn write_output(&mut self, output: QueryOutput) -> Result<()> {
        match output {
            QueryOutput::Scan { batches } | QueryOutput::Aggregate { batches, .. } => {
                for batch in batches {
                    self.write_parquet_batch(&batch)?;
                }
                Ok(())
            }
            QueryOutput::Explain { .. } => Err(DodamError::UnsupportedSql(
                "COPY TO does not support EXPLAIN output".to_string(),
            )),
        }
    }

    fn write_parquet_batch(&mut self, batch: &RecordBatch) -> Result<()> {
        if self.primitive_writer.is_some() {
            return Err(DodamError::UnsupportedSql(
                "cannot mix RecordBatch writes after primitive Parquet writer started".to_string(),
            ));
        }
        if self.profile_enabled {
            self.stats.batches += 1;
            self.stats.rows += batch.num_rows();
        }
        if batch.num_rows() == 0 {
            return Ok(());
        }
        if self.writer.is_none() {
            let writer_started = self.profile_enabled.then(Instant::now);
            let file = File::create(&self.path)?;
            let buffer_size = parquet_buffer_size_bytes(self.buffer_size_override, batch);
            let properties = parquet_writer_properties_for_batch(self.options, batch);
            self.writer = Some(ArrowWriter::try_new(
                BufWriter::with_capacity(buffer_size, file),
                batch.schema(),
                Some(properties),
            )?);
            if let Some(writer_started) = writer_started {
                let elapsed = writer_started.elapsed();
                self.stats.column_prepare += elapsed;
                self.stats.writer_open += elapsed;
            }
        }
        if batch.num_rows() == 0 {
            return Ok(());
        }

        let write_started = self.profile_enabled.then(Instant::now);
        self.writer
            .as_mut()
            .expect("Parquet writer is initialized")
            .write(batch)?;
        if let Some(write_started) = write_started {
            let elapsed = write_started.elapsed();
            self.stats.write += elapsed;
            self.stats.arrow_write += elapsed;
        }
        Ok(())
    }

    fn finish_parquet(&mut self) -> Result<()> {
        let flush_started = self.profile_enabled.then(Instant::now);
        if let Some(writer) = self.primitive_writer.take() {
            writer.close()?;
            if let Some(flush_started) = flush_started {
                let elapsed = flush_started.elapsed();
                self.stats.flush += elapsed;
                self.stats.writer_close += elapsed;
            }
            if self.profile_enabled
                && let Ok(metadata) = std::fs::metadata(&self.path)
            {
                self.stats.bytes = metadata.len() as usize;
            }
            return Ok(());
        }
        let Some(writer) = self.writer.take() else {
            File::create(&self.path)?;
            return Ok(());
        };
        writer.close()?;
        if let Some(flush_started) = flush_started {
            let elapsed = flush_started.elapsed();
            self.stats.flush += elapsed;
            self.stats.writer_close += elapsed;
        }
        if self.profile_enabled
            && let Ok(metadata) = std::fs::metadata(&self.path)
        {
            self.stats.bytes = metadata.len() as usize;
        }
        Ok(())
    }
}

fn parquet_buffer_size_bytes(override_value: Option<usize>, batch: &RecordBatch) -> usize {
    override_value
        .or_else(copy_buffer_size_env)
        .filter(|value| *value > 0)
        .unwrap_or_else(|| {
            if batch.num_columns() > 2 {
                PARQUET_WIDE_SINK_BUFFER_BYTES
            } else {
                PARQUET_SINK_BUFFER_BYTES
            }
        })
}

fn primitive_parquet_sink_enabled(batch: &PrimitiveBatch) -> bool {
    if std::env::var("DODAM_ENABLE_PRIMITIVE_PARQUET_SINK")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
    {
        return true;
    }
    if std::env::var("DODAM_DISABLE_PRIMITIVE_PARQUET_SINK_AUTO")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
    {
        return false;
    }
    choose_primitive_parquet_sink(PrimitiveParquetSinkCostInput {
        supported: primitive_parquet_columns(batch).is_some(),
        rows: batch.num_rows(),
        min_rows: primitive_parquet_sink_auto_min_rows(),
    })
}

fn primitive_parquet_sink_auto_min_rows() -> usize {
    std::env::var("DODAM_PRIMITIVE_PARQUET_SINK_AUTO_MIN_ROWS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(64 * 1024)
}

fn primitive_parquet_buffer_size_bytes(
    override_value: Option<usize>,
    batch: &PrimitiveBatch,
) -> usize {
    override_value
        .or_else(copy_buffer_size_env)
        .filter(|value| *value > 0)
        .unwrap_or_else(|| {
            if batch.columns.len() > 2 {
                PARQUET_WIDE_SINK_BUFFER_BYTES
            } else {
                PARQUET_SINK_BUFFER_BYTES
            }
        })
}

struct PrimitiveParquetFileWriter {
    writer: SerializedFileWriter<BufWriter<File>>,
    columns: Vec<PrimitiveParquetColumn>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PrimitiveParquetColumn {
    I32,
    I64,
}

impl PrimitiveParquetFileWriter {
    fn try_new(
        path: &Path,
        buffer_size: usize,
        options: ParquetCopyOptions,
        batch: &PrimitiveBatch,
    ) -> Result<Option<Self>> {
        let Some(columns) = primitive_parquet_columns(batch) else {
            return Ok(None);
        };
        if !primitive_parquet_options_supported(options) {
            return Ok(None);
        }
        let schema = primitive_parquet_schema(batch, &columns)?;
        let file = File::create(path)?;
        let properties = primitive_parquet_writer_properties(options, batch);
        let writer = SerializedFileWriter::new(
            BufWriter::with_capacity(buffer_size, file),
            schema,
            Arc::new(properties),
        )?;
        Ok(Some(Self { writer, columns }))
    }

    fn write_batch(&mut self, batch: PrimitiveBatch) -> Result<()> {
        let Some(columns) = primitive_parquet_columns(&batch) else {
            return Err(DodamError::UnsupportedSql(
                "primitive Parquet writer received unsupported batch".to_string(),
            ));
        };
        if columns != self.columns {
            return Err(DodamError::UnsupportedSql(
                "primitive Parquet writer schema changed".to_string(),
            ));
        }
        if batch.is_empty() {
            return Ok(());
        }
        let mut row_group = self.writer.next_row_group()?;
        for (column, column_type) in batch.columns.into_iter().zip(self.columns.iter()) {
            let Some(mut writer) = row_group.next_column()? else {
                return Err(DodamError::UnsupportedSql(
                    "primitive Parquet row group ended early".to_string(),
                ));
            };
            match (column.values, column_type) {
                (PrimitiveColumnValues::I32(values), PrimitiveParquetColumn::I32) => {
                    writer
                        .typed::<Int32Type>()
                        .write_batch(&values, None, None)?;
                }
                (PrimitiveColumnValues::I64(values), PrimitiveParquetColumn::I64) => {
                    writer
                        .typed::<Int64Type>()
                        .write_batch(&values, None, None)?;
                }
                _ => {
                    return Err(DodamError::UnsupportedSql(
                        "primitive Parquet column type mismatch".to_string(),
                    ));
                }
            }
            writer.close()?;
        }
        if row_group.next_column()?.is_some() {
            return Err(DodamError::UnsupportedSql(
                "primitive Parquet row group has extra columns".to_string(),
            ));
        }
        row_group.close()?;
        Ok(())
    }

    fn close(self) -> Result<()> {
        self.writer.close()?;
        Ok(())
    }
}

fn primitive_parquet_writer_properties(
    options: ParquetCopyOptions,
    batch: &PrimitiveBatch,
) -> WriterProperties {
    let mut builder = WriterProperties::builder()
        .set_compression(options.compression)
        .set_dictionary_enabled(false)
        .set_statistics_enabled(EnabledStatistics::Chunk)
        .set_max_row_group_row_count(options.max_row_group_rows)
        .set_write_batch_size(options.write_batch_size)
        .set_data_page_row_count_limit(options.data_page_row_count_limit);

    if parquet_auto_delta_encoding_enabled() {
        builder = add_primitive_batch_delta_encoding(builder, batch);
    }

    builder.build()
}

fn add_primitive_batch_delta_encoding(
    mut builder: WriterPropertiesBuilder,
    batch: &PrimitiveBatch,
) -> WriterPropertiesBuilder {
    for column in &batch.columns {
        let should_delta = match &column.values {
            PrimitiveColumnValues::I32(values) => {
                primitive_i32_values_monotonic_or_adjacent_duplicate(values)
            }
            PrimitiveColumnValues::I64(values) => {
                primitive_i64_values_monotonic_or_adjacent_duplicate(values)
            }
        };
        if should_delta {
            builder = builder.set_column_encoding(
                ColumnPath::from(column.name.as_str()),
                Encoding::DELTA_BINARY_PACKED,
            );
        }
    }
    builder
}

fn primitive_i32_values_monotonic_or_adjacent_duplicate(values: &[i32]) -> bool {
    if values.len() < 2 {
        return false;
    }
    let mut nondecreasing = true;
    let mut nonincreasing = true;
    let mut adjacent_duplicates = 0usize;
    let mut previous = values[0];
    for &value in &values[1..] {
        nondecreasing &= previous <= value;
        nonincreasing &= previous >= value;
        adjacent_duplicates += usize::from(previous == value);
        previous = value;
    }
    nondecreasing || nonincreasing || adjacent_duplicates * 2 >= values.len()
}

fn primitive_i64_values_monotonic_or_adjacent_duplicate(values: &[i64]) -> bool {
    if values.len() < 2 {
        return false;
    }
    let mut nondecreasing = true;
    let mut nonincreasing = true;
    let mut adjacent_duplicates = 0usize;
    let mut previous = values[0];
    for &value in &values[1..] {
        nondecreasing &= previous <= value;
        nonincreasing &= previous >= value;
        adjacent_duplicates += usize::from(previous == value);
        previous = value;
    }
    nondecreasing || nonincreasing || adjacent_duplicates * 2 >= values.len()
}

fn primitive_parquet_options_supported(options: ParquetCopyOptions) -> bool {
    !options.dictionary_enabled
}

fn primitive_parquet_columns(batch: &PrimitiveBatch) -> Option<Vec<PrimitiveParquetColumn>> {
    if batch.columns.is_empty() || batch.columns.iter().any(|column| column.nullable) {
        return None;
    }
    batch
        .columns
        .iter()
        .map(|column| match (&column.data_type, &column.values) {
            (DataType::Int32, PrimitiveColumnValues::I32(_))
            | (DataType::Date32, PrimitiveColumnValues::I32(_)) => {
                Some(PrimitiveParquetColumn::I32)
            }
            (DataType::Int64, PrimitiveColumnValues::I64(_)) => Some(PrimitiveParquetColumn::I64),
            _ => None,
        })
        .collect()
}

fn primitive_parquet_schema(
    batch: &PrimitiveBatch,
    columns: &[PrimitiveParquetColumn],
) -> Result<TypePtr> {
    let mut schema = String::from("message schema {\n");
    for (column, parquet_type) in batch.columns.iter().zip(columns.iter()) {
        let name = primitive_parquet_field_name(&column.name)?;
        match (column.data_type.clone(), parquet_type) {
            (DataType::Date32, PrimitiveParquetColumn::I32) => {
                schema.push_str(&format!("  REQUIRED INT32 {name} (DATE);\n"));
            }
            (_, PrimitiveParquetColumn::I32) => {
                schema.push_str(&format!("  REQUIRED INT32 {name};\n"));
            }
            (_, PrimitiveParquetColumn::I64) => {
                schema.push_str(&format!("  REQUIRED INT64 {name};\n"));
            }
        }
    }
    schema.push_str("}\n");
    Ok(Arc::new(parse_message_type(&schema)?))
}

fn primitive_parquet_field_name(name: &str) -> Result<&str> {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return Err(DodamError::UnsupportedSql(
            "empty primitive Parquet field name".to_string(),
        ));
    };
    if !(first == '_' || first.is_ascii_alphabetic())
        || !chars.all(|ch| ch == '_' || ch.is_ascii_alphanumeric())
    {
        return Err(DodamError::UnsupportedSql(format!(
            "primitive Parquet field name {name} is not supported by lower-level writer"
        )));
    }
    Ok(name)
}

fn parquet_writer_properties_for_batch(
    options: ParquetCopyOptions,
    first_batch: &RecordBatch,
) -> WriterProperties {
    let mut builder = WriterProperties::builder()
        .set_compression(options.compression)
        .set_dictionary_enabled(options.dictionary_enabled)
        .set_statistics_enabled(EnabledStatistics::Chunk)
        .set_max_row_group_row_count(options.max_row_group_rows)
        .set_write_batch_size(options.write_batch_size)
        .set_data_page_row_count_limit(options.data_page_row_count_limit);

    if parquet_auto_delta_encoding_enabled() {
        builder = add_monotonic_primitive_delta_encoding(builder, first_batch);
    }

    builder.build()
}

fn parquet_auto_delta_encoding_enabled() -> bool {
    !std::env::var("DODAM_DISABLE_PARQUET_AUTO_DELTA_ENCODING")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn add_monotonic_primitive_delta_encoding(
    mut builder: WriterPropertiesBuilder,
    batch: &RecordBatch,
) -> WriterPropertiesBuilder {
    for (field, column) in batch.schema().fields().iter().zip(batch.columns()) {
        if !matches!(
            field.data_type(),
            DataType::Int32 | DataType::Int64 | DataType::Date32
        ) {
            continue;
        }
        if primitive_column_is_monotonic_or_adjacent_duplicate(column.as_ref()) {
            builder = builder.set_column_encoding(
                ColumnPath::from(field.name().as_str()),
                Encoding::DELTA_BINARY_PACKED,
            );
        }
    }
    builder
}

fn primitive_column_is_monotonic_or_adjacent_duplicate(column: &dyn Array) -> bool {
    if column.len() < 2 || column.null_count() > 0 {
        return false;
    }
    if let Some(array) = column.as_any().downcast_ref::<Int32Array>() {
        return i32_array_is_monotonic_or_adjacent_duplicate(array);
    }
    if let Some(array) = column.as_any().downcast_ref::<Int64Array>() {
        return i64_array_is_monotonic_or_adjacent_duplicate(array);
    }
    false
}

fn i32_array_is_monotonic_or_adjacent_duplicate(array: &Int32Array) -> bool {
    let mut nondecreasing = true;
    let mut nonincreasing = true;
    let mut adjacent_duplicates = 0usize;
    let mut previous = array.value(0);
    for index in 1..array.len() {
        let value = array.value(index);
        nondecreasing &= previous <= value;
        nonincreasing &= previous >= value;
        adjacent_duplicates += usize::from(previous == value);
        previous = value;
    }
    nondecreasing || nonincreasing || adjacent_duplicates * 2 >= array.len()
}

fn i64_array_is_monotonic_or_adjacent_duplicate(array: &Int64Array) -> bool {
    let mut nondecreasing = true;
    let mut nonincreasing = true;
    let mut adjacent_duplicates = 0usize;
    let mut previous = array.value(0);
    for index in 1..array.len() {
        let value = array.value(index);
        nondecreasing &= previous <= value;
        nonincreasing &= previous >= value;
        adjacent_duplicates += usize::from(previous == value);
        previous = value;
    }
    nondecreasing || nonincreasing || adjacent_duplicates * 2 >= array.len()
}

impl RecordBatchSink for ParquetFileQuerySink {
    fn write_batch(&mut self, batch: &RecordBatch) -> Result<()> {
        self.write_parquet_batch(batch)
    }

    fn write_primitive_batch(&mut self, batch: PrimitiveBatch) -> Result<bool> {
        if let Some(writer) = self.primitive_writer.as_mut() {
            if self.profile_enabled {
                self.stats.batches += 1;
                self.stats.rows += batch.num_rows();
            }
            let write_started = self.profile_enabled.then(Instant::now);
            writer.write_batch(batch)?;
            if let Some(write_started) = write_started {
                let elapsed = write_started.elapsed();
                self.stats.write += elapsed;
                self.stats.primitive_write += elapsed;
            }
            return Ok(true);
        }
        if self.writer.is_none() && primitive_parquet_sink_enabled(&batch) && !batch.is_empty() {
            let writer_started = self.profile_enabled.then(Instant::now);
            let buffer_size =
                primitive_parquet_buffer_size_bytes(self.buffer_size_override, &batch);
            if let Some(mut writer) =
                PrimitiveParquetFileWriter::try_new(&self.path, buffer_size, self.options, &batch)?
            {
                if let Some(writer_started) = writer_started {
                    let elapsed = writer_started.elapsed();
                    self.stats.column_prepare += elapsed;
                    self.stats.writer_open += elapsed;
                }
                if self.profile_enabled {
                    self.stats.batches += 1;
                    self.stats.rows += batch.num_rows();
                }
                let write_started = self.profile_enabled.then(Instant::now);
                writer.write_batch(batch)?;
                if let Some(write_started) = write_started {
                    let elapsed = write_started.elapsed();
                    self.stats.write += elapsed;
                    self.stats.primitive_write += elapsed;
                }
                self.primitive_writer = Some(writer);
                return Ok(true);
            }
        }
        let schema = match &self.primitive_schema {
            Some(schema) if batch.matches_schema(schema.as_ref()) => schema.clone(),
            _ => {
                let schema = batch.schema();
                self.primitive_schema = Some(schema.clone());
                schema
            }
        };
        let convert_started = self.profile_enabled.then(Instant::now);
        let batch = batch.into_record_batch_with_schema(schema)?;
        if let Some(convert_started) = convert_started {
            self.stats.primitive_to_record_batch += convert_started.elapsed();
        }
        self.write_parquet_batch(&batch)?;
        Ok(true)
    }

    fn finish(&mut self) -> Result<()> {
        self.finish_parquet()
    }
}

#[derive(Debug, Default)]
pub struct CsvSinkStats {
    pub batches: usize,
    pub rows: usize,
    pub bytes: usize,
    pub header: Duration,
    pub column_prepare: Duration,
    pub serialize: Duration,
    pub write: Duration,
    pub flush: Duration,
    pub writer_open: Duration,
    pub writer_close: Duration,
    pub primitive_to_record_batch: Duration,
    pub arrow_write: Duration,
    pub primitive_write: Duration,
}

pub struct CopyProfile {
    pub enabled: bool,
    pub command_started: Instant,
    pub copy_parse: Duration,
    pub sink_create: Duration,
    pub direct_sink: Option<Duration>,
    pub streaming: Option<Duration>,
    pub materialize: Option<Duration>,
    pub write_output: Option<Duration>,
    pub finish: Option<Duration>,
    pub scan_plan_metrics: Option<ScanPlanMetrics>,
}

impl CopyProfile {
    pub fn new(enabled: bool, command_started: Instant) -> Self {
        Self {
            enabled,
            command_started,
            copy_parse: Duration::ZERO,
            sink_create: Duration::ZERO,
            direct_sink: None,
            streaming: None,
            materialize: None,
            write_output: None,
            finish: None,
            scan_plan_metrics: None,
        }
    }

    pub fn print(&self, sink: &CsvSinkStats) {
        if !self.enabled {
            return;
        }
        eprintln!(
            "copy_profile total={}us copy_parse={}us sink_create={}us direct_sink={} streaming={} materialize={} write_output={} finish={}",
            micros(self.command_started.elapsed()),
            micros(self.copy_parse),
            micros(self.sink_create),
            optional_micros(self.direct_sink),
            optional_micros(self.streaming),
            optional_micros(self.materialize),
            optional_micros(self.write_output),
            optional_micros(self.finish),
        );
        eprintln!(
            "copy_profile_sink batches={} rows={} bytes={} header={}us column_prepare={}us serialize={}us write={}us flush={}us writer_open={}us writer_close={}us primitive_to_record_batch={}us arrow_write={}us primitive_write={}us",
            sink.batches,
            sink.rows,
            sink.bytes,
            micros(sink.header),
            micros(sink.column_prepare),
            micros(sink.serialize),
            micros(sink.write),
            micros(sink.flush),
            micros(sink.writer_open),
            micros(sink.writer_close),
            micros(sink.primitive_to_record_batch),
            micros(sink.arrow_write),
            micros(sink.primitive_write),
        );
        if let Some(metrics) = self.scan_plan_metrics {
            eprintln!(
                "copy_profile_engine metadata={}us planning={}us decode={}us filter={}us projection={}us join_build={}us join_materialize={}us join_rows build={} probe={} output={}",
                nanos_to_micros(metrics.metadata_nanos),
                nanos_to_micros(metrics.planning_nanos),
                nanos_to_micros(metrics.decode_nanos),
                nanos_to_micros(metrics.filter_nanos),
                nanos_to_micros(metrics.projection_nanos),
                nanos_to_micros(metrics.join_build_nanos),
                nanos_to_micros(metrics.join_materialize_nanos),
                metrics.join_build_rows,
                metrics.join_probe_rows,
                metrics.join_output_rows,
            );
        }
    }
}

fn optional_micros(duration: Option<Duration>) -> String {
    duration
        .map(|duration| format!("{}us", micros(duration)))
        .unwrap_or_else(|| "n/a".to_string())
}

fn micros(duration: Duration) -> u128 {
    duration.as_micros()
}

fn nanos_to_micros(nanos: u64) -> u64 {
    nanos / 1_000
}

impl RecordBatchSink for CsvFileQuerySink {
    fn write_batch(&mut self, batch: &RecordBatch) -> Result<()> {
        self.write_csv_batch(batch)
    }

    fn supports_i32_utf8_rows(&self) -> bool {
        true
    }

    fn write_i32_utf8_rows(
        &mut self,
        left: &Int32Array,
        left_indices: &[u32],
        right_arrays: &[&StringArray],
        right_batch_indices: &[usize],
        right_row_indices: &[u32],
    ) -> Result<bool> {
        if left_indices.len() != right_batch_indices.len()
            || left_indices.len() != right_row_indices.len()
        {
            return Err(DodamError::UnsupportedSql(
                "mismatched direct CSV join row indices".to_string(),
            ));
        }
        if self.header && !self.wrote_header {
            let header_started = self.profile_enabled.then(Instant::now);
            self.writer.write_all(b"id,payload\n")?;
            if let Some(header_started) = header_started {
                self.stats.bytes += b"id,payload\n".len();
                self.stats.header += header_started.elapsed();
            }
            self.wrote_header = true;
        }
        if self.profile_enabled {
            self.stats.batches += 1;
            self.stats.rows += left_indices.len();
        }
        if self.discard {
            return Ok(true);
        }

        let serialize_started = self.profile_enabled.then(Instant::now);
        self.buffer.clear();
        self.buffer.reserve(left_indices.len().saturating_mul(24));
        let mut int_buffer = itoa::Buffer::new();
        let no_nulls =
            left.null_count() == 0 && right_arrays.iter().all(|array| array.null_count() == 0);
        if no_nulls {
            for ((&left_index, &right_batch), &right_row) in left_indices
                .iter()
                .zip(right_batch_indices)
                .zip(right_row_indices)
            {
                self.buffer.extend_from_slice(
                    int_buffer
                        .format(left.value(left_index as usize))
                        .as_bytes(),
                );
                self.buffer.push(b',');
                let right = right_arrays.get(right_batch).ok_or_else(|| {
                    DodamError::UnsupportedSql(
                        "join build row reference is out of range".to_string(),
                    )
                })?;
                self.buffer
                    .extend_from_slice(right.value(right_row as usize).as_bytes());
                self.buffer.push(b'\n');
            }
        } else {
            for ((&left_index, &right_batch), &right_row) in left_indices
                .iter()
                .zip(right_batch_indices)
                .zip(right_row_indices)
            {
                let left_index = left_index as usize;
                if !left.is_null(left_index) {
                    self.buffer
                        .extend_from_slice(int_buffer.format(left.value(left_index)).as_bytes());
                }
                self.buffer.push(b',');
                let right = right_arrays.get(right_batch).ok_or_else(|| {
                    DodamError::UnsupportedSql(
                        "join build row reference is out of range".to_string(),
                    )
                })?;
                let right_row = right_row as usize;
                if !right.is_null(right_row) {
                    self.buffer
                        .extend_from_slice(right.value(right_row).as_bytes());
                }
                self.buffer.push(b'\n');
            }
        }
        if let Some(serialize_started) = serialize_started {
            self.stats.serialize += serialize_started.elapsed();
        }

        let write_started = self.profile_enabled.then(Instant::now);
        let bytes = self.buffer.len();
        self.writer.write_all(&self.buffer)?;
        if let Some(write_started) = write_started {
            self.stats.bytes += bytes;
            self.stats.write += write_started.elapsed();
        }
        Ok(true)
    }

    fn supports_i32_rows(&self) -> bool {
        true
    }

    fn write_i32_rows(&mut self, array: &Int32Array, indices: &[u32]) -> Result<bool> {
        if self.header && !self.wrote_header {
            let header_started = self.profile_enabled.then(Instant::now);
            self.writer.write_all(b"id\n")?;
            if let Some(header_started) = header_started {
                self.stats.bytes += b"id\n".len();
                self.stats.header += header_started.elapsed();
            }
            self.wrote_header = true;
        }
        if self.profile_enabled {
            self.stats.batches += 1;
            self.stats.rows += indices.len();
        }
        if self.discard {
            return Ok(true);
        }

        let serialize_started = self.profile_enabled.then(Instant::now);
        self.buffer.clear();
        self.buffer.reserve(indices.len().saturating_mul(12));
        let mut int_buffer = itoa::Buffer::new();
        if array.null_count() == 0 {
            for &index in indices {
                self.buffer
                    .extend_from_slice(int_buffer.format(array.value(index as usize)).as_bytes());
                self.buffer.push(b'\n');
            }
        } else {
            for &index in indices {
                let index = index as usize;
                if !array.is_null(index) {
                    self.buffer
                        .extend_from_slice(int_buffer.format(array.value(index)).as_bytes());
                }
                self.buffer.push(b'\n');
            }
        }
        if let Some(serialize_started) = serialize_started {
            self.stats.serialize += serialize_started.elapsed();
        }

        let write_started = self.profile_enabled.then(Instant::now);
        let bytes = self.buffer.len();
        self.writer.write_all(&self.buffer)?;
        if let Some(write_started) = write_started {
            self.stats.bytes += bytes;
            self.stats.write += write_started.elapsed();
        }
        Ok(true)
    }

    fn discards_output(&self) -> bool {
        self.discard
    }

    fn finish(&mut self) -> Result<()> {
        self.finish_csv()
    }
}

fn copy_buffer_size_bytes(cli_value: Option<usize>, default_value: usize) -> usize {
    cli_value
        .or_else(copy_buffer_size_env)
        .filter(|value| *value > 0)
        .unwrap_or(default_value)
}

fn copy_buffer_size_env() -> Option<usize> {
    std::env::var("DODAM_COPY_BUFFER_BYTES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
}

pub fn write_csv_record_batch<W: Write>(batch: &RecordBatch, writer: &mut W) -> Result<()> {
    if batch.num_rows() == 0 {
        return Ok(());
    }
    let columns = batch
        .columns()
        .iter()
        .map(|array| CsvColumn::try_from_array(array.as_ref(), batch.num_rows()))
        .collect::<Result<Vec<_>>>()?;
    let mut buffer = Vec::with_capacity(
        batch
            .num_rows()
            .saturating_mul(columns.len())
            .saturating_mul(16),
    );
    if !write_specialized_csv_batch(&columns, batch.num_rows(), &mut buffer) {
        for row in 0..batch.num_rows() {
            for (column_index, column) in columns.iter().enumerate() {
                if column_index > 0 {
                    buffer.push(b',');
                }
                column.write_value(row, &mut buffer)?;
            }
            buffer.push(b'\n');
        }
    }
    writer.write_all(&buffer)?;
    Ok(())
}

enum CsvColumn<'a> {
    Int32(&'a Int32Array),
    Int64(&'a Int64Array),
    UInt32(&'a UInt32Array),
    UInt64(&'a UInt64Array),
    Float64(&'a Float64Array),
    Utf8 {
        array: &'a StringArray,
        quote_mode: Utf8QuoteMode,
    },
    Boolean(&'a BooleanArray),
}

#[derive(Debug, Clone, Copy)]
enum Utf8QuoteMode {
    Never,
    CheckEachValue,
}

impl<'a> CsvColumn<'a> {
    fn try_from_array(array: &'a dyn Array, rows: usize) -> Result<Self> {
        if let Some(array) = array.as_any().downcast_ref::<Int32Array>() {
            Ok(Self::Int32(array))
        } else if let Some(array) = array.as_any().downcast_ref::<Int64Array>() {
            Ok(Self::Int64(array))
        } else if let Some(array) = array.as_any().downcast_ref::<UInt32Array>() {
            Ok(Self::UInt32(array))
        } else if let Some(array) = array.as_any().downcast_ref::<UInt64Array>() {
            Ok(Self::UInt64(array))
        } else if let Some(array) = array.as_any().downcast_ref::<Float64Array>() {
            Ok(Self::Float64(array))
        } else if let Some(array) = array.as_any().downcast_ref::<StringArray>() {
            Ok(Self::Utf8 {
                array,
                quote_mode: utf8_quote_mode(array, rows),
            })
        } else if let Some(array) = array.as_any().downcast_ref::<BooleanArray>() {
            Ok(Self::Boolean(array))
        } else {
            Err(DodamError::UnsupportedSql(format!(
                "COPY CSV does not support {} columns yet",
                array.data_type()
            )))
        }
    }

    fn write_value(&self, row: usize, output: &mut Vec<u8>) -> Result<()> {
        match self {
            Self::Int32(array) => {
                if !array.is_null(row) {
                    let mut buffer = itoa::Buffer::new();
                    output.extend_from_slice(buffer.format(array.value(row)).as_bytes());
                }
            }
            Self::Int64(array) => {
                if !array.is_null(row) {
                    let mut buffer = itoa::Buffer::new();
                    output.extend_from_slice(buffer.format(array.value(row)).as_bytes());
                }
            }
            Self::UInt32(array) => {
                if !array.is_null(row) {
                    let mut buffer = itoa::Buffer::new();
                    output.extend_from_slice(buffer.format(array.value(row)).as_bytes());
                }
            }
            Self::UInt64(array) => {
                if !array.is_null(row) {
                    let mut buffer = itoa::Buffer::new();
                    output.extend_from_slice(buffer.format(array.value(row)).as_bytes());
                }
            }
            Self::Float64(array) => {
                if !array.is_null(row) {
                    let mut buffer = ryu::Buffer::new();
                    output.extend_from_slice(buffer.format(array.value(row)).as_bytes());
                }
            }
            Self::Utf8 { array, quote_mode } => {
                if !array.is_null(row) {
                    let value = array.value(row).as_bytes();
                    match quote_mode {
                        Utf8QuoteMode::Never => output.extend_from_slice(value),
                        Utf8QuoteMode::CheckEachValue => write_csv_bytes(value, output),
                    }
                }
            }
            Self::Boolean(array) => {
                if !array.is_null(row) {
                    output.extend_from_slice(if array.value(row) { b"true" } else { b"false" });
                }
            }
        }
        Ok(())
    }
}

fn write_specialized_csv_batch(
    columns: &[CsvColumn<'_>],
    rows: usize,
    output: &mut Vec<u8>,
) -> bool {
    match columns {
        [CsvColumn::Int32(array)] => {
            write_i32_csv_batch(array, rows, output);
            true
        }
        [
            CsvColumn::Int32(left),
            CsvColumn::Utf8 {
                array: right,
                quote_mode: Utf8QuoteMode::Never,
            },
        ] => {
            write_i32_utf8_no_quote_csv_batch(left, right, rows, output);
            true
        }
        [
            CsvColumn::Int32(left),
            CsvColumn::Utf8 {
                array: right,
                quote_mode: Utf8QuoteMode::CheckEachValue,
            },
        ] => {
            write_i32_utf8_csv_batch(left, right, rows, output);
            true
        }
        [CsvColumn::Int32(left), rest @ ..]
            if !rest.is_empty()
                && rest.iter().all(|column| {
                    matches!(
                        column,
                        CsvColumn::Utf8 {
                            quote_mode: Utf8QuoteMode::Never,
                            ..
                        }
                    )
                }) =>
        {
            let utf8_columns = rest
                .iter()
                .map(|column| match column {
                    CsvColumn::Utf8 { array, .. } => *array,
                    _ => unreachable!("all rest columns are no-quote Utf8"),
                })
                .collect::<Vec<_>>();
            write_i32_utf8_columns_no_quote_csv_batch(left, &utf8_columns, rows, output);
            true
        }
        _ => false,
    }
}

fn write_i32_csv_batch(array: &Int32Array, rows: usize, output: &mut Vec<u8>) {
    let mut int_buffer = itoa::Buffer::new();
    for row in 0..rows {
        if !array.is_null(row) {
            output.extend_from_slice(int_buffer.format(array.value(row)).as_bytes());
        }
        output.push(b'\n');
    }
}

fn write_i32_utf8_no_quote_csv_batch(
    left: &Int32Array,
    right: &StringArray,
    rows: usize,
    output: &mut Vec<u8>,
) {
    let mut int_buffer = itoa::Buffer::new();
    if left.null_count() == 0 && right.null_count() == 0 {
        for row in 0..rows {
            output.extend_from_slice(int_buffer.format(left.value(row)).as_bytes());
            output.push(b',');
            output.extend_from_slice(right.value(row).as_bytes());
            output.push(b'\n');
        }
        return;
    }

    for row in 0..rows {
        if !left.is_null(row) {
            output.extend_from_slice(int_buffer.format(left.value(row)).as_bytes());
        }
        output.push(b',');
        if !right.is_null(row) {
            output.extend_from_slice(right.value(row).as_bytes());
        }
        output.push(b'\n');
    }
}

fn write_i32_utf8_csv_batch(
    left: &Int32Array,
    right: &StringArray,
    rows: usize,
    output: &mut Vec<u8>,
) {
    let mut int_buffer = itoa::Buffer::new();
    for row in 0..rows {
        if !left.is_null(row) {
            output.extend_from_slice(int_buffer.format(left.value(row)).as_bytes());
        }
        output.push(b',');
        if !right.is_null(row) {
            write_csv_bytes(right.value(row).as_bytes(), output);
        }
        output.push(b'\n');
    }
}

fn write_i32_utf8_columns_no_quote_csv_batch(
    left: &Int32Array,
    utf8_columns: &[&StringArray],
    rows: usize,
    output: &mut Vec<u8>,
) {
    let mut int_buffer = itoa::Buffer::new();
    let no_nulls =
        left.null_count() == 0 && utf8_columns.iter().all(|array| array.null_count() == 0);
    if no_nulls {
        for row in 0..rows {
            output.extend_from_slice(int_buffer.format(left.value(row)).as_bytes());
            for array in utf8_columns {
                output.push(b',');
                output.extend_from_slice(array.value(row).as_bytes());
            }
            output.push(b'\n');
        }
        return;
    }

    for row in 0..rows {
        if !left.is_null(row) {
            output.extend_from_slice(int_buffer.format(left.value(row)).as_bytes());
        }
        for array in utf8_columns {
            output.push(b',');
            if !array.is_null(row) {
                output.extend_from_slice(array.value(row).as_bytes());
            }
        }
        output.push(b'\n');
    }
}

fn utf8_quote_mode(array: &StringArray, rows: usize) -> Utf8QuoteMode {
    for row in 0..rows {
        if !array.is_null(row) && csv_bytes_need_quote(array.value(row).as_bytes()) {
            return Utf8QuoteMode::CheckEachValue;
        }
    }
    Utf8QuoteMode::Never
}

fn write_csv_bytes(value: &[u8], output: &mut Vec<u8>) {
    if !csv_bytes_need_quote(value) {
        output.extend_from_slice(value);
        return;
    }

    output.push(b'"');
    for byte in value {
        if *byte == b'"' {
            output.extend_from_slice(b"\"\"");
        } else {
            output.push(*byte);
        }
    }
    output.push(b'"');
}

fn csv_bytes_need_quote(value: &[u8]) -> bool {
    value
        .iter()
        .any(|byte| matches!(byte, b',' | b'"' | b'\n' | b'\r'))
}
