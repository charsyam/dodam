use std::fs::{self, File};
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::time::Instant;

use arrow::record_batch::RecordBatch;
use clap::Parser;
use dodam::engine::DodamEngine;
use dodam::error::{DodamError, Result};
use dodam::sql::{QueryOutput, execute_sql};
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, Encoding};
use parquet::file::properties::WriterProperties;
use parquet::schema::types::ColumnPath;

const DEFAULT_BATCH_SIZE: usize = 16 * 1024;
const DEFAULT_OUTPUT_DIR: &str = "/tmp/dodam-tpch-inprocess";

const TABLES: &[&str] = &[
    "lineitem", "orders", "customer", "supplier", "partsupp", "part", "nation", "region",
];

const KEYWORDS: &[&str] = &[
    "and", "as", "cross", "from", "full", "group", "having", "inner", "join", "left", "limit",
    "on", "or", "order", "outer", "right", "select", "semi", "where",
];

#[derive(Debug, Parser)]
#[command(about = "Run the canonical TPC-H inventory queries in one Dodam process")]
struct Args {
    #[arg(long, default_value = "/tmp/dodam-tpchgen-sf1")]
    data_dir: PathBuf,

    #[arg(long, default_value = "tests/tpch_coverage.rs")]
    queries: PathBuf,

    #[arg(long, default_value = DEFAULT_OUTPUT_DIR)]
    output_dir: PathBuf,

    #[arg(long, default_value_t = DEFAULT_BATCH_SIZE)]
    batch_size: usize,

    #[arg(long)]
    show_sql: bool,

    #[arg(long, default_value_t = 1)]
    repeats: usize,

    #[arg(long, value_delimiter = ',')]
    only: Vec<String>,
}

#[derive(Debug)]
struct TpchQuery {
    name: String,
    sql: String,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    configure_default_rayon_threads();
    let args = Args::parse();
    validate_data_dir(&args.data_dir)?;
    fs::create_dir_all(&args.output_dir)?;

    let queries = filter_queries(load_tpch_queries(&args.queries)?, &args.only)?;
    let engine = DodamEngine::default();
    let mut total = 0.0;
    let mut ok = 0usize;

    for repeat in 0..args.repeats {
        let mut repeat_total = 0.0;
        let mut repeat_ok = 0usize;
        for query in &queries {
            let sql = rewrite_table_refs(&query.sql, &args.data_dir);
            if args.show_sql {
                println!("\n-- {}\n{}\n", query.name, sql);
            }
            let output_path =
                args.output_dir
                    .join(format!("r{}_{}.parquet", repeat + 1, query.name));
            let started = Instant::now();
            let result = execute_sql(&engine, &sql, args.batch_size).await;
            match result {
                Ok(output) => {
                    let (rows, batches) = write_parquet_output(&output_path, output)?;
                    let elapsed = started.elapsed().as_secs_f64();
                    total += elapsed;
                    repeat_total += elapsed;
                    ok += 1;
                    repeat_ok += 1;
                    let cache_stats = engine.file_cache_stats();
                    println!(
                        "repeat={} {}: ok {:.6}s rows={} batches={} metadata_cache={} file_cache={} file_cache_bytes={} cache_hits={} cache_misses={} cache_evictions={} cache_read_bytes={} cache_deferred_admissions={}",
                        repeat + 1,
                        query.name,
                        elapsed,
                        rows,
                        batches,
                        engine.metadata_cache_len(),
                        engine.file_cache_len(),
                        engine.file_cache_bytes(),
                        cache_stats.hits,
                        cache_stats.misses,
                        cache_stats.evictions,
                        cache_stats.read_bytes,
                        cache_stats.deferred_admissions
                    );
                }
                Err(error) => {
                    let elapsed = started.elapsed().as_secs_f64();
                    println!(
                        "repeat={} {}: fail {:.6}s {}",
                        repeat + 1,
                        query.name,
                        elapsed,
                        error
                    );
                }
            }
        }
        let cache_stats = engine.file_cache_stats();
        println!(
            "repeat={} summary: {}/{} ok total={:.6}s metadata_cache={} file_cache={} file_cache_bytes={} cache_hits={} cache_misses={} cache_evictions={} cache_read_bytes={} cache_deferred_admissions={}",
            repeat + 1,
            repeat_ok,
            queries.len(),
            repeat_total,
            engine.metadata_cache_len(),
            engine.file_cache_len(),
            engine.file_cache_bytes(),
            cache_stats.hits,
            cache_stats.misses,
            cache_stats.evictions,
            cache_stats.read_bytes,
            cache_stats.deferred_admissions
        );
    }

    let cache_stats = engine.file_cache_stats();
    println!(
        "TPC-H in-process over {}: {}/{} ok total={:.6}s metadata_cache={} file_cache={} file_cache_bytes={} cache_hits={} cache_misses={} cache_evictions={} cache_read_bytes={} cache_deferred_admissions={}",
        args.data_dir.display(),
        ok,
        queries.len().saturating_mul(args.repeats),
        total,
        engine.metadata_cache_len(),
        engine.file_cache_len(),
        engine.file_cache_bytes(),
        cache_stats.hits,
        cache_stats.misses,
        cache_stats.evictions,
        cache_stats.read_bytes,
        cache_stats.deferred_admissions
    );
    Ok(())
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

fn validate_data_dir(data_dir: &Path) -> Result<()> {
    for table in TABLES {
        let path = data_dir.join(format!("{table}.parquet"));
        if !path.exists() {
            return Err(DodamError::MissingPath(path));
        }
    }
    Ok(())
}

fn load_tpch_queries(path: &Path) -> Result<Vec<TpchQuery>> {
    let source = fs::read_to_string(path)?;
    let mut queries = Vec::new();
    for block in source.split("TpchQuery {").skip(1) {
        let Some(name) = extract_between(block, "name: \"", "\",") else {
            continue;
        };
        let Some(sql) = extract_between(block, "sql: r#\"", "\"#,") else {
            continue;
        };
        queries.push(TpchQuery {
            name: name.to_string(),
            sql: sql.to_string(),
        });
    }
    Ok(queries)
}

fn filter_queries(mut queries: Vec<TpchQuery>, filters: &[String]) -> Result<Vec<TpchQuery>> {
    if filters.is_empty() {
        return Ok(queries);
    }
    let filters = filters
        .iter()
        .map(|filter| filter.to_ascii_lowercase())
        .collect::<Vec<_>>();
    queries.retain(|query| {
        let name = query.name.to_ascii_lowercase();
        filters
            .iter()
            .any(|filter| name == *filter || name.contains(filter.as_str()))
    });
    if queries.is_empty() {
        return Err(DodamError::UnsupportedSql(format!(
            "no TPC-H queries matched --only {}",
            filters.join(",")
        )));
    }
    Ok(queries)
}

fn extract_between<'a>(input: &'a str, prefix: &str, suffix: &str) -> Option<&'a str> {
    let start = input.find(prefix)? + prefix.len();
    let end = input[start..].find(suffix)? + start;
    Some(&input[start..end])
}

fn write_parquet_output(path: &Path, output: QueryOutput) -> Result<(usize, usize)> {
    let batches = match output {
        QueryOutput::Scan { batches } | QueryOutput::Aggregate { batches, .. } => batches,
        QueryOutput::Explain { .. } => {
            return Err(DodamError::UnsupportedSql(
                "TPC-H in-process runner does not support EXPLAIN output".to_string(),
            ));
        }
    };
    let rows = batches.iter().map(RecordBatch::num_rows).sum();
    let batch_count = batches.len();
    write_batches_to_parquet(path, &batches)?;
    Ok((rows, batch_count))
}

fn write_batches_to_parquet(path: &Path, batches: &[RecordBatch]) -> Result<()> {
    if batches.is_empty() {
        File::create(path)?;
        return Ok(());
    }
    let file = File::create(path)?;
    let properties = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_dictionary_enabled(false)
        .set_max_row_group_row_count(Some(64 * 1024))
        .set_write_batch_size(8 * 1024)
        .set_data_page_row_count_limit(8 * 1024)
        .set_column_encoding(ColumnPath::from("f.id"), Encoding::DELTA_BINARY_PACKED)
        .set_column_encoding(ColumnPath::from("id"), Encoding::DELTA_BINARY_PACKED)
        .build();
    let mut writer = ArrowWriter::try_new(
        BufWriter::with_capacity(8 * 1024 * 1024, file),
        batches[0].schema(),
        Some(properties),
    )?;
    for batch in batches {
        if batch.num_rows() > 0 {
            writer.write(batch)?;
        }
    }
    writer.close()?;
    Ok(())
}

fn rewrite_table_refs(sql: &str, data_dir: &Path) -> String {
    let tokens = tokenize_sql(sql);
    let mut output = String::with_capacity(sql.len() + 256);
    let mut index = 0usize;
    let mut expect_table = false;

    while index < tokens.len() {
        match &tokens[index] {
            Token::Word(word) if expect_table && is_table(word) => {
                let table = word.to_ascii_lowercase();
                let (alias, next_index) = consume_alias(&tokens, index + 1, &table);
                output.push('\'');
                output.push_str(
                    &data_dir
                        .join(format!("{table}.parquet"))
                        .display()
                        .to_string(),
                );
                output.push_str("' AS ");
                output.push_str(&alias);
                index = next_index;
                expect_table = false;
                continue;
            }
            Token::Word(word) if expect_table => {
                expect_table = false;
                output.push_str(word);
            }
            Token::Word(word) => {
                let lower = word.to_ascii_lowercase();
                if lower == "from" || lower == "join" {
                    expect_table = true;
                }
                output.push_str(word);
            }
            Token::Punct(',') => {
                expect_table = true;
                output.push(',');
            }
            token => output.push_str(token.as_str()),
        }
        index += 1;
    }

    output.trim().to_string()
}

fn consume_alias(tokens: &[Token], mut index: usize, table: &str) -> (String, usize) {
    let original_index = index;
    index = skip_whitespace(tokens, index);
    if matches_word(tokens.get(index), "as") {
        let alias_index = skip_whitespace(tokens, index + 1);
        if let Some(Token::Word(alias)) = tokens.get(alias_index) {
            return (alias.clone(), alias_index + 1);
        }
        return (table.to_string(), index);
    }
    if let Some(Token::Word(alias)) = tokens.get(index)
        && !is_keyword(alias)
    {
        return (alias.clone(), index + 1);
    }
    (table.to_string(), original_index)
}

fn skip_whitespace(tokens: &[Token], mut index: usize) -> usize {
    while matches!(tokens.get(index), Some(Token::Whitespace(_))) {
        index += 1;
    }
    index
}

fn matches_word(token: Option<&Token>, expected: &str) -> bool {
    matches!(token, Some(Token::Word(word)) if word.eq_ignore_ascii_case(expected))
}

fn is_table(word: &str) -> bool {
    let lower = word.to_ascii_lowercase();
    TABLES.contains(&lower.as_str())
}

fn is_keyword(word: &str) -> bool {
    let lower = word.to_ascii_lowercase();
    KEYWORDS.contains(&lower.as_str())
}

#[derive(Debug, Clone)]
enum Token {
    Word(String),
    Whitespace(String),
    Quoted(String),
    Punct(char),
    Other(String),
}

impl Token {
    fn as_str(&self) -> &str {
        match self {
            Self::Word(value)
            | Self::Whitespace(value)
            | Self::Quoted(value)
            | Self::Other(value) => value,
            Self::Punct(',') => ",",
            Self::Punct('(') => "(",
            Self::Punct(')') => ")",
            Self::Punct(';') => ";",
            Self::Punct(_) => "",
        }
    }
}

fn tokenize_sql(sql: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = sql.chars().collect();
    let mut index = 0usize;
    while index < chars.len() {
        let ch = chars[index];
        if ch.is_whitespace() {
            let start = index;
            index += 1;
            while index < chars.len() && chars[index].is_whitespace() {
                index += 1;
            }
            tokens.push(Token::Whitespace(chars[start..index].iter().collect()));
        } else if ch == '\'' {
            let start = index;
            index += 1;
            while index < chars.len() {
                if chars[index] == '\'' {
                    index += 1;
                    if index < chars.len() && chars[index] == '\'' {
                        index += 1;
                        continue;
                    }
                    break;
                }
                index += 1;
            }
            tokens.push(Token::Quoted(chars[start..index].iter().collect()));
        } else if ch.is_ascii_alphanumeric() || ch == '_' {
            let start = index;
            index += 1;
            while index < chars.len()
                && (chars[index].is_ascii_alphanumeric() || chars[index] == '_')
            {
                index += 1;
            }
            tokens.push(Token::Word(chars[start..index].iter().collect()));
        } else if matches!(ch, ',' | '(' | ')' | ';') {
            tokens.push(Token::Punct(ch));
            index += 1;
        } else {
            tokens.push(Token::Other(ch.to_string()));
            index += 1;
        }
    }
    tokens
}
