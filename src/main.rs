use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use arrow::array::{
    Array, BooleanArray, Float64Array, Int32Array, Int64Array, StringArray, UInt32Array,
    UInt64Array,
};
use arrow::record_batch::RecordBatch;
use clap::{Parser, Subcommand};
use dodam::catalog::PersistentCatalog;
use dodam::engine::DodamEngine;
use dodam::error::{DodamError, Result};
use dodam::execution::{
    AggregateExpr, AggregateMetrics, FilterExpr, Projection, RecordBatchSink, ScanPlanMetrics,
    SortExpr,
};
use dodam::sql::{QueryOutput, execute_sql, try_execute_sql_streaming, try_execute_sql_to_sink};
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, Encoding};
use parquet::file::properties::WriterProperties;
use parquet::schema::types::ColumnPath;
use sqlparser::ast::{CopyOption, CopySource, CopyTarget, Ident, Statement};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser as SqlParser;

const DEFAULT_BATCH_SIZE: usize = 16 * 1024;

#[derive(Debug, Parser)]
#[command(name = "dodam")]
#[command(about = "A vectorized OLAP engine built around Iceberg-style planning and Parquet scans")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Manage the local .dodam catalog.
    Catalog {
        #[command(subcommand)]
        command: CatalogCommands,
    },

    /// Scan a Parquet file or a directory containing Parquet files.
    Scan {
        /// Parquet file or directory path.
        path: PathBuf,

        /// Number of rows per Arrow RecordBatch.
        #[arg(long, default_value_t = DEFAULT_BATCH_SIZE)]
        batch_size: usize,

        /// Optional row limit for quick inspection.
        #[arg(long)]
        limit: Option<usize>,

        /// Comma-separated column list to project.
        #[arg(long, value_delimiter = ',')]
        columns: Vec<String>,

        /// Equality filter in column=value form.
        #[arg(long)]
        filter: Option<String>,

        /// Optional column or "column desc" expression for deterministic ordering before limit.
        #[arg(long)]
        order_by: Option<String>,

        /// Sort descending when --order-by only names a column.
        #[arg(long)]
        desc: bool,
    },

    /// Compute global aggregates over a Parquet file or directory.
    Aggregate {
        /// Parquet file or directory path.
        path: PathBuf,

        /// Number of rows per Arrow RecordBatch.
        #[arg(long, default_value_t = DEFAULT_BATCH_SIZE)]
        batch_size: usize,

        /// Aggregate expression, such as count(*), count(id), sum(id), avg(id), min(id), or max(id).
        #[arg(long = "agg", required = true)]
        aggregates: Vec<String>,

        /// Optional filter expression.
        #[arg(long)]
        filter: Option<String>,

        /// Comma-separated group key columns.
        #[arg(long, value_delimiter = ',')]
        group_by: Vec<String>,
    },

    /// Execute a supported SQL SELECT query.
    Query {
        /// SQL query text.
        sql: String,

        /// Number of rows per Arrow RecordBatch.
        #[arg(long, default_value_t = DEFAULT_BATCH_SIZE)]
        batch_size: usize,

        /// COPY TO output buffer size in bytes.
        #[arg(long)]
        copy_buffer_size: Option<usize>,
    },
}

#[derive(Debug, Subcommand)]
enum CatalogCommands {
    /// Register a local Parquet file or directory as a table.
    Register {
        /// Table name.
        name: String,

        /// Local Parquet file or directory path.
        path: PathBuf,
    },

    /// Refresh a registered table snapshot from its stored location.
    Refresh {
        /// Table name.
        name: String,
    },

    /// List registered tables.
    List,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    configure_default_rayon_threads();
    let cli = Cli::parse();
    let engine = DodamEngine::default();

    match cli.command {
        Commands::Catalog { command } => {
            let catalog = PersistentCatalog::new(std::env::current_dir()?);
            match command {
                CatalogCommands::Register { name, path } => {
                    let entry = catalog.register_local_parquet(name, path)?;
                    println!(
                        "registered table {} at {} ({:?})",
                        entry.name, entry.location, entry.format
                    );
                }
                CatalogCommands::Refresh { name } => {
                    let entry = catalog.refresh_table(&name)?;
                    let fragments = entry
                        .metadata
                        .as_ref()
                        .map(|metadata| metadata.statistics.fragments)
                        .unwrap_or_default();
                    println!(
                        "refreshed table {} at {} ({:?}, {} fragment(s))",
                        entry.name, entry.location, entry.format, fragments
                    );
                }
                CatalogCommands::List => {
                    let tables = catalog.tables()?;
                    if tables.is_empty() {
                        println!("no tables registered");
                    } else {
                        for table in tables {
                            println!("{}\t{:?}\t{}", table.name, table.format, table.location);
                        }
                    }
                }
            }
        }
        Commands::Scan {
            path,
            batch_size,
            limit,
            columns,
            filter,
            order_by,
            desc,
        } => {
            let projection = if columns.is_empty() {
                Projection::All
            } else {
                Projection::Columns(columns)
            };
            let filter = filter.as_deref().map(FilterExpr::parse).transpose()?;
            let metrics = if let Some(order_by) = order_by {
                let mut order_by = SortExpr::parse(&order_by)?;
                order_by.descending |= desc;
                engine
                    .scan_parquet_ordered(path, batch_size, limit, projection, filter, order_by)
                    .await?
            } else {
                engine
                    .scan_parquet(path, batch_size, limit, projection, filter)
                    .await?
            };
            println!(
                "scanned {} rows x {} column(s) in {} batches from {} fragment(s); row groups: {} scanned, {} pruned, {} total; compressed bytes: {} scanned, {} pruned, {} total; timing: metadata={}us planning={}us decode={}us filter={}us projection={}us limit={}us",
                metrics.rows,
                metrics.columns,
                metrics.batches,
                metrics.fragments,
                metrics.row_groups_scanned,
                metrics.row_groups_pruned,
                metrics.row_groups_total,
                metrics.compressed_bytes_scanned,
                metrics.compressed_bytes_pruned,
                metrics.compressed_bytes_total,
                nanos_to_micros(metrics.metadata_nanos),
                nanos_to_micros(metrics.planning_nanos),
                nanos_to_micros(metrics.decode_nanos),
                nanos_to_micros(metrics.filter_nanos),
                nanos_to_micros(metrics.projection_nanos),
                nanos_to_micros(metrics.limit_nanos),
            );
        }
        Commands::Aggregate {
            path,
            batch_size,
            aggregates,
            filter,
            group_by,
        } => {
            let aggregates = aggregates
                .iter()
                .map(|aggregate| AggregateExpr::parse(aggregate))
                .collect::<Result<Vec<_>>>()?;
            let filter = filter.as_deref().map(FilterExpr::parse).transpose()?;
            if group_by.is_empty() {
                let metrics = engine
                    .aggregate_parquet(path, batch_size, aggregates, filter)
                    .await?;
                let values = metrics
                    .values
                    .iter()
                    .map(|value| format!("{}={}", value.expr, value.value))
                    .collect::<Vec<_>>()
                    .join(", ");
                println!(
                    "aggregated {} rows in {} batches from {} fragment(s) aggregate={:.3}ms merge={:.3}ms: {}",
                    metrics.rows,
                    metrics.batches,
                    metrics.fragments,
                    nanos_to_millis(metrics.aggregate_nanos),
                    nanos_to_millis(metrics.aggregate_merge_nanos),
                    values
                );
            } else {
                let metrics = engine
                    .aggregate_parquet_grouped(path, batch_size, aggregates, group_by, filter)
                    .await?;
                println!(
                    "aggregated {} rows into {} group(s) in {} batches from {} fragment(s) aggregate={:.3}ms merge={:.3}ms",
                    metrics.rows,
                    metrics.groups.len(),
                    metrics.batches,
                    metrics.fragments,
                    nanos_to_millis(metrics.aggregate_nanos),
                    nanos_to_millis(metrics.aggregate_merge_nanos)
                );
                for group in metrics.groups {
                    let keys = group
                        .keys
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join(", ");
                    let values = group
                        .values
                        .iter()
                        .map(|value| format!("{}={}", value.expr, value.value))
                        .collect::<Vec<_>>()
                        .join(", ");
                    println!("group [{keys}]: {values}");
                }
            }
        }
        Commands::Query {
            sql,
            batch_size,
            copy_buffer_size,
        } => {
            let command_started = Instant::now();
            let profile_enabled = copy_profile_enabled();
            let query_profile_enabled = query_profile_enabled();
            let copy_parse_started = Instant::now();
            let copy = parse_copy_to_select(&sql)?;
            let copy_parse_elapsed = copy_parse_started.elapsed();
            if let Some(copy) = copy {
                let mut profile = CopyProfile::new(profile_enabled, command_started);
                profile.copy_parse = copy_parse_elapsed;
                let sink_started = Instant::now();
                let mut sink = CopyFileQuerySink::new(
                    &copy.path,
                    copy.format,
                    copy.header,
                    copy.parquet_options,
                    copy_buffer_size,
                    profile_enabled,
                )?;
                profile.sink_create = sink_started.elapsed();

                if copy_sql_may_use_direct_or_streaming(&copy.sql) {
                    let direct_started = Instant::now();
                    if let Some(metrics) =
                        try_execute_sql_to_sink(&engine, &copy.sql, batch_size, &mut sink).await?
                    {
                        profile.direct_sink = Some(direct_started.elapsed());
                        profile.scan_plan_metrics = Some(metrics);
                        profile.print(sink.stats());
                        return Ok(());
                    }
                    profile.direct_sink = Some(direct_started.elapsed());

                    let streaming_started = Instant::now();
                    if let Some(stream) =
                        try_execute_sql_streaming(&engine, &copy.sql, batch_size).await?
                    {
                        engine.write_batches_to_sink(stream, &mut sink)?;
                        profile.streaming = Some(streaming_started.elapsed());
                        profile.print(sink.stats());
                        return Ok(());
                    }
                    profile.streaming = Some(streaming_started.elapsed());
                } else {
                    profile.direct_sink = Some(Duration::ZERO);
                    profile.streaming = Some(Duration::ZERO);
                }

                let materialize_started = Instant::now();
                let output = execute_sql(&engine, &copy.sql, batch_size).await?;
                profile.materialize = Some(materialize_started.elapsed());
                let write_output_started = Instant::now();
                sink.write_output(output)?;
                profile.write_output = Some(write_output_started.elapsed());
                let finish_started = Instant::now();
                sink.finish()?;
                profile.finish = Some(finish_started.elapsed());
                profile.print(sink.stats());
                return Ok(());
            }

            let mut sink = StdoutQuerySink;
            let direct_started = Instant::now();
            if try_execute_sql_to_sink(&engine, &sql, batch_size, &mut sink)
                .await?
                .is_some()
            {
                print_query_profile(
                    query_profile_enabled,
                    command_started,
                    copy_parse_elapsed,
                    Some(direct_started.elapsed()),
                    None,
                    None,
                    None,
                );
                return Ok(());
            }
            let direct_elapsed = direct_started.elapsed();
            let streaming_started = Instant::now();
            if let Some(stream) = try_execute_sql_streaming(&engine, &sql, batch_size).await? {
                engine.write_batches_to_sink(stream, &mut sink)?;
                print_query_profile(
                    query_profile_enabled,
                    command_started,
                    copy_parse_elapsed,
                    Some(direct_elapsed),
                    Some(streaming_started.elapsed()),
                    None,
                    None,
                );
                return Ok(());
            }
            let streaming_elapsed = streaming_started.elapsed();
            let execute_started = Instant::now();
            let output = execute_sql(&engine, &sql, batch_size).await?;
            let execute_elapsed = execute_started.elapsed();
            let write_started = Instant::now();
            sink.write_output(output)?;
            print_query_profile(
                query_profile_enabled,
                command_started,
                copy_parse_elapsed,
                Some(direct_elapsed),
                Some(streaming_elapsed),
                Some(execute_elapsed),
                Some(write_started.elapsed()),
            );
        }
    }

    Ok(())
}

struct CopyToSelect {
    sql: String,
    path: PathBuf,
    header: bool,
    format: CopyFormat,
    parquet_options: ParquetCopyOptions,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CopyFormat {
    Csv,
    Parquet,
}

#[derive(Debug, Clone, Copy)]
struct ParquetCopyOptions {
    compression: Compression,
    dictionary_enabled: bool,
    max_row_group_rows: Option<usize>,
    write_batch_size: usize,
    data_page_row_count_limit: usize,
}

impl Default for ParquetCopyOptions {
    fn default() -> Self {
        Self {
            compression: Compression::SNAPPY,
            dictionary_enabled: false,
            max_row_group_rows: Some(64 * 1024),
            write_batch_size: 8 * 1024,
            data_page_row_count_limit: 8 * 1024,
        }
    }
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
    clean_copy_option_value(value)
        .parse::<usize>()
        .map_err(|_| {
            DodamError::UnsupportedSql(format!("{option_name} expects a positive integer"))
        })
}

fn parse_copy_to_select(sql: &str) -> Result<Option<CopyToSelect>> {
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

fn ident_eq(ident: &Ident, expected: &str) -> bool {
    ident.value.eq_ignore_ascii_case(expected)
}

struct CsvFileQuerySink {
    writer: BufWriter<File>,
    buffer: Vec<u8>,
    wrote_header: bool,
    header: bool,
    discard: bool,
    profile_enabled: bool,
    stats: CsvSinkStats,
}

enum CopyFileQuerySink {
    Csv(CsvFileQuerySink),
    Parquet(ParquetFileQuerySink),
}

impl CopyFileQuerySink {
    fn new(
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

    fn stats(&self) -> &CsvSinkStats {
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

struct ParquetFileQuerySink {
    path: PathBuf,
    writer: Option<ArrowWriter<BufWriter<File>>>,
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
            let properties = WriterProperties::builder()
                .set_compression(self.options.compression)
                .set_dictionary_enabled(self.options.dictionary_enabled)
                .set_max_row_group_row_count(self.options.max_row_group_rows)
                .set_write_batch_size(self.options.write_batch_size)
                .set_data_page_row_count_limit(self.options.data_page_row_count_limit)
                .set_column_encoding(ColumnPath::from("f.id"), Encoding::DELTA_BINARY_PACKED)
                .set_column_encoding(ColumnPath::from("id"), Encoding::DELTA_BINARY_PACKED);
            let properties = properties.build();
            self.writer = Some(ArrowWriter::try_new(
                BufWriter::with_capacity(buffer_size, file),
                batch.schema(),
                Some(properties),
            )?);
            if let Some(writer_started) = writer_started {
                self.stats.column_prepare += writer_started.elapsed();
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
            self.stats.write += write_started.elapsed();
        }
        Ok(())
    }

    fn finish_parquet(&mut self) -> Result<()> {
        let flush_started = self.profile_enabled.then(Instant::now);
        let Some(writer) = self.writer.take() else {
            File::create(&self.path)?;
            return Ok(());
        };
        writer.close()?;
        if let Some(flush_started) = flush_started {
            self.stats.flush += flush_started.elapsed();
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

impl RecordBatchSink for ParquetFileQuerySink {
    fn write_batch(&mut self, batch: &RecordBatch) -> Result<()> {
        self.write_parquet_batch(batch)
    }

    fn finish(&mut self) -> Result<()> {
        self.finish_parquet()
    }
}

#[derive(Debug, Default)]
struct CsvSinkStats {
    batches: usize,
    rows: usize,
    bytes: usize,
    header: Duration,
    column_prepare: Duration,
    serialize: Duration,
    write: Duration,
    flush: Duration,
}

struct CopyProfile {
    enabled: bool,
    command_started: Instant,
    copy_parse: Duration,
    sink_create: Duration,
    direct_sink: Option<Duration>,
    streaming: Option<Duration>,
    materialize: Option<Duration>,
    write_output: Option<Duration>,
    finish: Option<Duration>,
    scan_plan_metrics: Option<ScanPlanMetrics>,
}

impl CopyProfile {
    fn new(enabled: bool, command_started: Instant) -> Self {
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

    fn print(&self, sink: &CsvSinkStats) {
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
            "copy_profile_sink batches={} rows={} bytes={} header={}us column_prepare={}us serialize={}us write={}us flush={}us",
            sink.batches,
            sink.rows,
            sink.bytes,
            micros(sink.header),
            micros(sink.column_prepare),
            micros(sink.serialize),
            micros(sink.write),
            micros(sink.flush),
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

fn copy_profile_enabled() -> bool {
    std::env::var("DODAM_PROFILE_COPY").is_ok_and(|value| {
        matches!(
            value.to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

fn query_profile_enabled() -> bool {
    std::env::var("DODAM_PROFILE_QUERY").is_ok_and(|value| {
        matches!(
            value.to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

fn copy_sql_may_use_direct_or_streaming(sql: &str) -> bool {
    let lower = sql.to_ascii_lowercase();
    if lower.contains(" group by ")
        || lower.contains(" order by ")
        || lower.contains(" having ")
        || lower.contains(" distinct ")
        || lower.contains(" exists")
        || lower.contains(" in (select")
        || lower.contains(" sum(")
        || lower.contains(" count(")
        || lower.contains(" avg(")
        || lower.contains(" min(")
        || lower.contains(" max(")
    {
        return false;
    }
    true
}

fn configure_default_rayon_threads() {
    if std::env::var_os("RAYON_NUM_THREADS").is_some() {
        return;
    }
    let threads = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
        .min(16)
        .max(1);
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global();
}

fn print_query_profile(
    enabled: bool,
    command_started: Instant,
    copy_parse: Duration,
    direct_sink: Option<Duration>,
    streaming: Option<Duration>,
    execute: Option<Duration>,
    write_output: Option<Duration>,
) {
    if !enabled {
        return;
    }
    eprintln!(
        "query_profile total={}us copy_parse={}us direct_sink={} streaming={} execute={} write_output={}",
        micros(command_started.elapsed()),
        micros(copy_parse),
        optional_micros(direct_sink),
        optional_micros(streaming),
        optional_micros(execute),
        optional_micros(write_output),
    );
}

fn optional_micros(duration: Option<Duration>) -> String {
    duration
        .map(|duration| format!("{}us", micros(duration)))
        .unwrap_or_else(|| "n/a".to_string())
}

fn micros(duration: Duration) -> u128 {
    duration.as_micros()
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

struct StdoutQuerySink;

impl StdoutQuerySink {
    fn write_output(&mut self, output: QueryOutput) -> Result<()> {
        match output {
            QueryOutput::Scan { batches } => self.write_batches(batches)?,
            QueryOutput::Aggregate { metrics, batches } => {
                self.write_batches(batches)?;
                if query_summary_enabled() {
                    self.write_aggregate_summary(&metrics);
                }
            }
            QueryOutput::Explain { plan } => println!("{plan}"),
        }
        Ok(())
    }

    fn write_batches(&mut self, batches: Vec<RecordBatch>) -> Result<()> {
        for batch in batches {
            self.write_batch(&batch)?;
        }
        Ok(())
    }

    fn write_stdout_batch(&mut self, batch: &RecordBatch) -> Result<()> {
        if batch.num_rows() == 0 {
            return Ok(());
        }
        let columns = match batch
            .columns()
            .iter()
            .map(|array| CsvColumn::try_from_array(array.as_ref(), batch.num_rows()))
            .collect::<Result<Vec<_>>>()
        {
            Ok(columns) => columns,
            Err(DodamError::UnsupportedSql(_)) => {
                println!("{batch:?}");
                return Ok(());
            }
            Err(error) => return Err(error),
        };
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
        std::io::stdout().write_all(&buffer)?;
        Ok(())
    }

    fn write_aggregate_summary(&mut self, metrics: &AggregateMetrics) {
        if metrics.groups.is_empty() {
            let values = metrics
                .values
                .iter()
                .map(|value| format!("{}={}", value.expr, value.value))
                .collect::<Vec<_>>()
                .join(", ");
            println!(
                "aggregated {} rows in {} batches from {} fragment(s) aggregate={:.3}ms merge={:.3}ms: {}",
                metrics.rows,
                metrics.batches,
                metrics.fragments,
                nanos_to_millis(metrics.aggregate_nanos),
                nanos_to_millis(metrics.aggregate_merge_nanos),
                values
            );
            return;
        }

        println!(
            "aggregated {} rows into {} group(s) in {} batches from {} fragment(s) aggregate={:.3}ms merge={:.3}ms",
            metrics.rows,
            metrics.groups.len(),
            metrics.batches,
            metrics.fragments,
            nanos_to_millis(metrics.aggregate_nanos),
            nanos_to_millis(metrics.aggregate_merge_nanos)
        );
        for group in &metrics.groups {
            let keys = group
                .keys
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            let values = group
                .values
                .iter()
                .map(|value| format!("{}={}", value.expr, value.value))
                .collect::<Vec<_>>()
                .join(", ");
            println!("group [{keys}]: {values}");
        }
    }
}

impl RecordBatchSink for StdoutQuerySink {
    fn write_batch(&mut self, batch: &RecordBatch) -> Result<()> {
        self.write_stdout_batch(batch)
    }
}

fn nanos_to_micros(nanos: u64) -> u64 {
    nanos / 1_000
}

fn nanos_to_millis(nanos: u64) -> f64 {
    nanos as f64 / 1_000_000.0
}

fn query_summary_enabled() -> bool {
    std::env::var("DODAM_QUERY_SUMMARY")
        .ok()
        .is_some_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}
