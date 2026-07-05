use std::path::PathBuf;

use parquet::basic::Compression;
use sqlparser::ast::{CopyOption, CopySource, CopyTarget, Ident, Statement};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser as SqlParser;

use crate::error::{DodamError, Result};

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
            max_row_group_rows: Some(64 * 1024),
            write_batch_size: 8 * 1024,
            data_page_row_count_limit: 8 * 1024,
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

fn ident_eq(ident: &Ident, expected: &str) -> bool {
    ident.value.eq_ignore_ascii_case(expected)
}
