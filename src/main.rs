use std::fs;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use arrow::record_batch::RecordBatch;
use clap::{Parser, Subcommand};
use dodam::catalog::PersistentCatalog;
use dodam::copy::{CopyFileQuerySink, CopyProfile, parse_copy_to_select, write_csv_record_batch};
use dodam::engine::DodamEngine;
use dodam::error::Result;
use dodam::execution::{
    AggregateExpr, AggregateMetrics, FilterExpr, Projection, RecordBatchSink, SortExpr,
};
use dodam::sql::{QueryOutput, SqlResultSink, SqlSinkExecutionOptions, execute_sql_to_result_sink};

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

    /// Execute SQL statements from a file in one process.
    QueryFile {
        /// File containing one or more SQL statements separated by semicolons.
        path: PathBuf,

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
            run_query_sql(&engine, &sql, batch_size, copy_buffer_size).await?;
        }
        Commands::QueryFile {
            path,
            batch_size,
            copy_buffer_size,
        } => {
            let contents = fs::read_to_string(path)?;
            let statements = split_sql_statements(&contents);
            for statement in statements {
                run_query_sql(&engine, &statement, batch_size, copy_buffer_size).await?;
            }
        }
    }

    Ok(())
}

async fn run_query_sql(
    engine: &DodamEngine,
    sql: &str,
    batch_size: usize,
    copy_buffer_size: Option<usize>,
) -> Result<()> {
    let command_started = Instant::now();
    let profile_enabled = copy_profile_enabled();
    let query_profile_enabled = query_profile_enabled();
    let copy_parse_started = Instant::now();
    let copy = parse_copy_to_select(sql)?;
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

        let execution = execute_sql_to_result_sink(
            engine,
            &copy.sql,
            batch_size,
            &mut sink,
            SqlSinkExecutionOptions::default(),
        )
        .await?;
        profile.direct_sink = execution.direct_sink;
        profile.streaming = execution.streaming;
        profile.materialize = execution.execute;
        profile.write_output = execution.write_output;
        profile.scan_plan_metrics = execution.scan_plan_metrics;
        let finish_started = Instant::now();
        sink.finish()?;
        profile.finish = Some(finish_started.elapsed());
        profile.print(sink.stats());
        return Ok(());
    }

    let mut sink = StdoutQuerySink;
    let execution = execute_sql_to_result_sink(
        engine,
        sql,
        batch_size,
        &mut sink,
        SqlSinkExecutionOptions::default(),
    )
    .await?;
    print_query_profile(
        query_profile_enabled,
        command_started,
        copy_parse_elapsed,
        execution.direct_sink,
        execution.streaming,
        execution.execute,
        execution.write_output,
    );
    Ok(())
}

fn split_sql_statements(sql: &str) -> Vec<String> {
    let mut statements = Vec::new();
    let mut start = 0usize;
    let mut in_quote = false;
    let bytes = sql.as_bytes();
    let mut index = 0usize;
    while index < bytes.len() {
        match bytes[index] {
            b'\'' => {
                if in_quote && bytes.get(index + 1) == Some(&b'\'') {
                    index += 1;
                } else {
                    in_quote = !in_quote;
                }
            }
            b';' if !in_quote => {
                let statement = sql[start..index].trim();
                if !statement.is_empty() {
                    statements.push(statement.to_string());
                }
                start = index + 1;
            }
            _ => {}
        }
        index += 1;
    }
    let statement = sql[start..].trim();
    if !statement.is_empty() {
        statements.push(statement.to_string());
    }
    statements
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
        match write_csv_record_batch(batch, &mut std::io::stdout()) {
            Ok(()) => Ok(()),
            Err(dodam::error::DodamError::UnsupportedSql(_)) => {
                println!("{batch:?}");
                return Ok(());
            }
            Err(error) => return Err(error),
        }
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

impl SqlResultSink for StdoutQuerySink {
    fn record_batch_sink(&mut self) -> &mut dyn RecordBatchSink {
        self
    }

    fn write_output(&mut self, output: QueryOutput) -> Result<()> {
        StdoutQuerySink::write_output(self, output)
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
