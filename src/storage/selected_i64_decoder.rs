use bytes::Bytes;
use parquet::basic::{Encoding, Type as ParquetPhysicalType};
use parquet::column::page::Page;

use crate::error::{DodamError, Result};

use super::{
    DirectPrimitiveColumnScanMetrics, advance_run_cursor, count_present_def_levels,
    decode_dictionary_indices_selected_nullable_ranges, decode_dictionary_indices_selected_ranges,
    decode_rle_i16_values, num_required_bits_i16, parse_v1_rle_level_data, runs_overlap_from,
};

pub(super) fn read_plain_i64_selected_runs(
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
    let mut def_levels = Vec::<i16>::new();
    let mut present_prefix = Vec::<usize>::new();
    let mut dictionary = None::<Vec<i64>>;
    let mut dictionary_ids = Vec::<i32>::new();
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
                dictionary = Some(decode_plain_i64_dictionary(buf, num_values as usize)?);
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
                    Encoding::PLAIN | Encoding::RLE_DICTIONARY | Encoding::PLAIN_DICTIONARY
                ) {
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
                    def_levels.clear();
                    decode_rle_i16_values(
                        level_data,
                        num_required_bits_i16(column_desc.max_def_level()),
                        page_rows,
                        &mut def_levels,
                    )?;
                    if encoding == Encoding::PLAIN {
                        copy_selected_i64_nullable_page(
                            &buf,
                            value_start,
                            page_row_start,
                            page_rows,
                            runs,
                            &def_levels,
                            column_desc.max_def_level(),
                            &mut present_prefix,
                            output,
                            metrics,
                        )?;
                    } else {
                        let values =
                            count_present_def_levels(&def_levels, column_desc.max_def_level());
                        append_dictionary_selected_i64_page(
                            buf.slice(value_start..),
                            values,
                            dictionary.as_deref(),
                            page_row_start,
                            page_rows,
                            runs,
                            Some((&def_levels, column_desc.max_def_level())),
                            &mut dictionary_ids,
                            output,
                            metrics,
                        )?;
                    }
                } else {
                    if encoding == Encoding::PLAIN {
                        copy_selected_i64_required_page(
                            &buf,
                            value_start,
                            page_row_start,
                            page_rows,
                            runs,
                            output,
                            metrics,
                        )?;
                    } else {
                        append_dictionary_selected_i64_page(
                            buf.slice(value_start..),
                            page_rows,
                            dictionary.as_deref(),
                            page_row_start,
                            page_rows,
                            runs,
                            None,
                            &mut dictionary_ids,
                            output,
                            metrics,
                        )?;
                    }
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
                    Encoding::PLAIN | Encoding::RLE_DICTIONARY | Encoding::PLAIN_DICTIONARY
                ) {
                    return Ok(None);
                }
                let page_rows = num_values as usize;
                let page_row_end = page_row_start + page_rows;
                advance_run_cursor(runs, &mut run_cursor, page_row_start);
                if !runs_overlap_from(runs, run_cursor, page_row_start, page_row_end) {
                    page_row_start = page_row_end;
                    continue;
                }
                let value_start = (rep_levels_byte_len + def_levels_byte_len) as usize;
                if value_start > buf.len() {
                    return Ok(None);
                }
                if column_desc.max_def_level() > 0 {
                    if num_nulls == 0 {
                        if encoding == Encoding::PLAIN {
                            copy_selected_i64_required_page(
                                &buf,
                                value_start,
                                page_row_start,
                                page_rows,
                                runs,
                                output,
                                metrics,
                            )?;
                        } else {
                            append_dictionary_selected_i64_page(
                                buf.slice(value_start..),
                                page_rows,
                                dictionary.as_deref(),
                                page_row_start,
                                page_rows,
                                runs,
                                None,
                                &mut dictionary_ids,
                                output,
                                metrics,
                            )?;
                        }
                    } else {
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
                        if encoding == Encoding::PLAIN {
                            copy_selected_i64_nullable_page(
                                &buf,
                                value_start,
                                page_row_start,
                                page_rows,
                                runs,
                                &def_levels,
                                column_desc.max_def_level(),
                                &mut present_prefix,
                                output,
                                metrics,
                            )?;
                        } else {
                            let values =
                                count_present_def_levels(&def_levels, column_desc.max_def_level());
                            append_dictionary_selected_i64_page(
                                buf.slice(value_start..),
                                values,
                                dictionary.as_deref(),
                                page_row_start,
                                page_rows,
                                runs,
                                Some((&def_levels, column_desc.max_def_level())),
                                &mut dictionary_ids,
                                output,
                                metrics,
                            )?;
                        }
                    }
                } else {
                    if encoding == Encoding::PLAIN {
                        copy_selected_i64_required_page(
                            &buf,
                            value_start,
                            page_row_start,
                            page_rows,
                            runs,
                            output,
                            metrics,
                        )?;
                    } else {
                        append_dictionary_selected_i64_page(
                            buf.slice(value_start..),
                            page_rows,
                            dictionary.as_deref(),
                            page_row_start,
                            page_rows,
                            runs,
                            None,
                            &mut dictionary_ids,
                            output,
                            metrics,
                        )?;
                    }
                }
                page_row_start = page_row_end;
            }
        }
    }
    if page_row_start != records {
        return Ok(None);
    }
    Ok(Some(()))
}

fn decode_plain_i64_dictionary(buf: Bytes, num_values: usize) -> Result<Vec<i64>> {
    let bytes = num_values.saturating_mul(std::mem::size_of::<i64>());
    if bytes > buf.len() {
        return Ok(Vec::new());
    }
    let mut values = Vec::with_capacity(num_values);
    for chunk in buf[..bytes].chunks_exact(8) {
        values.push(i64::from_le_bytes([
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
        ]));
    }
    Ok(values)
}

#[allow(clippy::too_many_arguments)]
fn append_dictionary_selected_i64_page(
    data: Bytes,
    values: usize,
    dictionary: Option<&[i64]>,
    page_row_start: usize,
    page_rows: usize,
    runs: &[(usize, usize)],
    nullable: Option<(&[i16], i16)>,
    ids: &mut Vec<i32>,
    output: &mut Vec<i64>,
    metrics: &mut DirectPrimitiveColumnScanMetrics,
) -> Result<()> {
    let Some(dictionary) = dictionary else {
        return Err(DodamError::UnsupportedSql(
            "dictionary selected i64 page missing dictionary".to_string(),
        ));
    };
    ids.clear();
    if let Some((def_levels, max_def_level)) = nullable {
        decode_dictionary_indices_selected_nullable_ranges(
            data,
            values,
            dictionary.len(),
            runs,
            page_row_start,
            page_rows,
            def_levels,
            max_def_level,
            ids,
        )?;
    } else {
        decode_dictionary_indices_selected_ranges(
            data,
            values,
            dictionary.len(),
            runs,
            page_row_start,
            page_rows,
            ids,
        )?;
    }
    metrics.add_selected_read(ids.len());
    output.reserve(ids.len());
    for id in ids.iter().copied() {
        let Ok(index) = usize::try_from(id) else {
            return Err(DodamError::UnsupportedSql(
                "selected nullable dictionary i64 contains null".to_string(),
            ));
        };
        let Some(value) = dictionary.get(index) else {
            return Err(DodamError::UnsupportedSql(
                "selected dictionary i64 id out of range".to_string(),
            ));
        };
        output.push(*value);
    }
    Ok(())
}

pub(super) fn read_plain_i64_selected_runs_sink<F>(
    row_group: &dyn parquet::file::reader::RowGroupReader,
    column: usize,
    records: usize,
    runs: &[(usize, usize)],
    mut consume: F,
) -> Result<Option<()>>
where
    F: FnMut(usize, &[i64]) -> Result<()>,
{
    let column_desc = row_group.metadata().schema_descr().column(column);
    if column_desc.physical_type() != ParquetPhysicalType::INT64
        || column_desc.max_rep_level() != 0
        || column_desc.max_def_level() > 1
    {
        return Ok(None);
    }
    let selected_prefix = selected_run_prefixes(runs);
    let mut page_reader = row_group.get_column_page_reader(column)?;
    let mut page_row_start = 0usize;
    let mut run_cursor = 0usize;
    let mut def_levels = Vec::<i16>::new();
    let mut present_prefix = Vec::<usize>::new();
    let mut scratch = Vec::<i64>::new();
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
                    def_levels.clear();
                    decode_rle_i16_values(
                        level_data,
                        num_required_bits_i16(column_desc.max_def_level()),
                        page_rows,
                        &mut def_levels,
                    )?;
                    copy_selected_i64_nullable_page_to_sink(
                        &buf,
                        value_start,
                        page_row_start,
                        page_rows,
                        runs,
                        &selected_prefix,
                        &def_levels,
                        column_desc.max_def_level(),
                        &mut present_prefix,
                        &mut scratch,
                        &mut consume,
                    )?;
                } else {
                    copy_selected_i64_required_page_to_sink(
                        &buf,
                        value_start,
                        page_row_start,
                        page_rows,
                        runs,
                        &selected_prefix,
                        &mut scratch,
                        &mut consume,
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
                let value_start = (rep_levels_byte_len + def_levels_byte_len) as usize;
                if value_start > buf.len() {
                    return Ok(None);
                }
                if column_desc.max_def_level() > 0 && num_nulls != 0 {
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
                    copy_selected_i64_nullable_page_to_sink(
                        &buf,
                        value_start,
                        page_row_start,
                        page_rows,
                        runs,
                        &selected_prefix,
                        &def_levels,
                        column_desc.max_def_level(),
                        &mut present_prefix,
                        &mut scratch,
                        &mut consume,
                    )?;
                } else {
                    copy_selected_i64_required_page_to_sink(
                        &buf,
                        value_start,
                        page_row_start,
                        page_rows,
                        runs,
                        &selected_prefix,
                        &mut scratch,
                        &mut consume,
                    )?;
                }
                page_row_start = page_row_end;
            }
        }
    }
    if page_row_start != records {
        return Ok(None);
    }
    Ok(Some(()))
}

fn selected_run_prefixes(runs: &[(usize, usize)]) -> Vec<usize> {
    let mut prefixes = Vec::with_capacity(runs.len());
    let mut selected = 0usize;
    for &(_, len) in runs {
        prefixes.push(selected);
        selected = selected.saturating_add(len);
    }
    prefixes
}

fn build_present_prefix(def_levels: &[i16], max_def_level: i16, present_prefix: &mut Vec<usize>) {
    present_prefix.clear();
    present_prefix.reserve(def_levels.len() + 1);
    present_prefix.push(0);
    let mut present = 0usize;
    for &level in def_levels {
        present += usize::from(level == max_def_level);
        present_prefix.push(present);
    }
}

fn copy_selected_i64_required_page(
    buf: &Bytes,
    value_start: usize,
    page_row_start: usize,
    page_rows: usize,
    runs: &[(usize, usize)],
    output: &mut Vec<i64>,
    metrics: &mut DirectPrimitiveColumnScanMetrics,
) -> Result<()> {
    let page_row_end = page_row_start + page_rows;
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
        copy_selected_i64_values(buf, byte_start, rows, output);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn copy_selected_i64_required_page_to_sink<F>(
    buf: &Bytes,
    value_start: usize,
    page_row_start: usize,
    page_rows: usize,
    runs: &[(usize, usize)],
    selected_prefix: &[usize],
    scratch: &mut Vec<i64>,
    consume: &mut F,
) -> Result<()>
where
    F: FnMut(usize, &[i64]) -> Result<()>,
{
    let page_row_end = page_row_start + page_rows;
    let value_bytes = page_rows.saturating_mul(std::mem::size_of::<i64>());
    if value_start.saturating_add(value_bytes) > buf.len() {
        return Err(DodamError::UnsupportedSql(
            "selected i64 page payload length mismatch".to_string(),
        ));
    }
    scratch.clear();
    let mut first_selected_offset = None;
    for (run_index, &(run_start, run_len)) in runs.iter().enumerate() {
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
        if first_selected_offset.is_none() {
            first_selected_offset =
                Some(selected_prefix[run_index] + local_start + page_row_start - run_start);
        }
        let byte_start = value_start + local_start * std::mem::size_of::<i64>();
        copy_selected_i64_values(buf, byte_start, rows, scratch);
    }
    if let Some(offset) = first_selected_offset
        && !scratch.is_empty()
    {
        consume(offset, scratch)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn copy_selected_i64_nullable_page(
    buf: &Bytes,
    value_start: usize,
    page_row_start: usize,
    page_rows: usize,
    runs: &[(usize, usize)],
    def_levels: &[i16],
    max_def_level: i16,
    present_prefix: &mut Vec<usize>,
    output: &mut Vec<i64>,
    metrics: &mut DirectPrimitiveColumnScanMetrics,
) -> Result<()> {
    if def_levels.len() != page_rows {
        return Ok(());
    }
    build_present_prefix(def_levels, max_def_level, present_prefix);
    let present_values = present_prefix[page_rows];
    let value_bytes = present_values.saturating_mul(std::mem::size_of::<i64>());
    if value_start.saturating_add(value_bytes) > buf.len() {
        return Err(DodamError::UnsupportedSql(
            "selected nullable i64 page payload length mismatch".to_string(),
        ));
    }
    let page_row_end = page_row_start + page_rows;
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
        if present_prefix[local_end] - present_prefix[local_start] != rows {
            return Ok(());
        }
        let present_before = present_prefix[local_start];
        let byte_start = value_start + present_before * std::mem::size_of::<i64>();
        metrics.add_selected_read(rows);
        copy_selected_i64_values(buf, byte_start, rows, output);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn copy_selected_i64_nullable_page_to_sink<F>(
    buf: &Bytes,
    value_start: usize,
    page_row_start: usize,
    page_rows: usize,
    runs: &[(usize, usize)],
    selected_prefix: &[usize],
    def_levels: &[i16],
    max_def_level: i16,
    present_prefix: &mut Vec<usize>,
    scratch: &mut Vec<i64>,
    consume: &mut F,
) -> Result<()>
where
    F: FnMut(usize, &[i64]) -> Result<()>,
{
    if def_levels.len() != page_rows {
        return Ok(());
    }
    build_present_prefix(def_levels, max_def_level, present_prefix);
    let present_values = present_prefix[page_rows];
    let value_bytes = present_values.saturating_mul(std::mem::size_of::<i64>());
    if value_start.saturating_add(value_bytes) > buf.len() {
        return Err(DodamError::UnsupportedSql(
            "selected nullable i64 page payload length mismatch".to_string(),
        ));
    }
    let page_row_end = page_row_start + page_rows;
    scratch.clear();
    let mut first_selected_offset = None;
    for (run_index, &(run_start, run_len)) in runs.iter().enumerate() {
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
        if present_prefix[local_end] - present_prefix[local_start] != rows {
            return Ok(());
        }
        if first_selected_offset.is_none() {
            first_selected_offset =
                Some(selected_prefix[run_index] + local_start + page_row_start - run_start);
        }
        let present_before = present_prefix[local_start];
        let byte_start = value_start + present_before * std::mem::size_of::<i64>();
        copy_selected_i64_values(buf, byte_start, rows, scratch);
    }
    if let Some(offset) = first_selected_offset
        && !scratch.is_empty()
    {
        consume(offset, scratch)?;
    }
    Ok(())
}

fn copy_selected_i64_values(buf: &Bytes, byte_start: usize, rows: usize, output: &mut Vec<i64>) {
    if rows == 0 {
        return;
    }
    #[cfg(target_endian = "little")]
    {
        if selected_i64_simd_copy_enabled() && copy_selected_i64_values_avx2_available() {
            unsafe {
                copy_selected_i64_values_avx2(buf, byte_start, rows, output);
            }
            return;
        }
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

#[cfg(target_arch = "x86_64")]
fn copy_selected_i64_values_avx2_available() -> bool {
    std::is_x86_feature_detected!("avx2")
}

#[cfg(not(target_arch = "x86_64"))]
fn copy_selected_i64_values_avx2_available() -> bool {
    false
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn copy_selected_i64_values_avx2(
    buf: &Bytes,
    byte_start: usize,
    rows: usize,
    output: &mut Vec<i64>,
) {
    use std::arch::x86_64::{__m256i, _mm256_loadu_si256, _mm256_storeu_si256};

    let before = output.len();
    output.reserve(rows);
    unsafe {
        output.set_len(before + rows);
    }
    let mut row = 0usize;
    while row + 4 <= rows {
        unsafe {
            let values = _mm256_loadu_si256(
                buf.as_ptr()
                    .add(byte_start + row * std::mem::size_of::<i64>())
                    .cast::<__m256i>(),
            );
            _mm256_storeu_si256(
                output.as_mut_ptr().add(before + row).cast::<__m256i>(),
                values,
            );
        }
        row += 4;
    }
    while row < rows {
        let offset = byte_start + row * std::mem::size_of::<i64>();
        unsafe {
            output[before + row] = i64::from_le_bytes([
                *buf.get_unchecked(offset),
                *buf.get_unchecked(offset + 1),
                *buf.get_unchecked(offset + 2),
                *buf.get_unchecked(offset + 3),
                *buf.get_unchecked(offset + 4),
                *buf.get_unchecked(offset + 5),
                *buf.get_unchecked(offset + 6),
                *buf.get_unchecked(offset + 7),
            ]);
        }
        row += 1;
    }
}

fn selected_i64_simd_copy_enabled() -> bool {
    std::env::var("DODAM_ENABLE_FUSED_SELECTED_I64_SIMD_COPY")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}
