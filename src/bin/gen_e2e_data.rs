use std::path::PathBuf;
use std::sync::Arc;

use arrow::array::{ArrayRef, Int32Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;

fn main() {
    let dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .expect("usage: gen_e2e_data <dir>");
    std::fs::create_dir_all(&dir).expect("create dir");
    write_facts_parquet(&dir.join("facts.parquet"), 262_144, 16_384);
    write_dim_parquet(&dir.join("dim.parquet"), 65_536, 16_384);
    write_sparse_facts_parquet(&dir.join("sparse-facts.parquet"), 262_144, 16_384);
    write_sparse_dim_parquet(&dir.join("sparse-dim.parquet"), 65_536, 16_384);
    write_duplicate_dim_parquet(&dir.join("duplicate-dim.parquet"), 131_072, 16_384);
    write_narrow_facts_parquet(&dir.join("narrow-facts.parquet"), 262_144, 16_384);
    write_wide_pruned_facts_parquet(&dir.join("wide-pruned-facts.parquet"), 262_144, 16_384);
    write_facts_parquet(&dir.join("small-rg-facts.parquet"), 262_144, 4_096);
    write_multi_facts_parquet(&dir.join("multi-facts.parquet"), 262_144, 16_384);
    write_multi_dim_parquet(&dir.join("multi-dim.parquet"), 65_536, 16_384);
    write_string_facts_parquet(&dir.join("string-facts.parquet"), 131_072, 16_384);
    write_string_dim_parquet(&dir.join("string-dim.parquet"), 32_768, 16_384);
    write_wide_facts_parquet(&dir.join("wide-facts.parquet"), 131_072, 16_384);
    write_wide_dim_parquet(&dir.join("wide-dim.parquet"), 32_768, 16_384);
}

fn write_facts_parquet(path: &std::path::Path, rows: usize, row_group_size: usize) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("key", DataType::Int32, false),
        Field::new("bucket", DataType::Utf8, false),
        Field::new("value", DataType::Int64, false),
        Field::new("payload", DataType::Utf8, false),
    ]));
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int32Array::from_iter_values(
            (0..rows).map(|value| value as i32),
        )),
        Arc::new(Int32Array::from_iter_values(
            (0..rows).map(|value| (value % 65_536) as i32),
        )),
        Arc::new(StringArray::from_iter_values(
            (0..rows).map(|value| format!("g{:04}", value % 1024)),
        )),
        Arc::new(Int64Array::from_iter_values(
            (0..rows).map(|value| value as i64 * 7),
        )),
        Arc::new(StringArray::from_iter_values(
            (0..rows).map(|value| format!("payload-{value}")),
        )),
    ];
    write_parquet(path, schema, columns, row_group_size);
}

fn write_dim_parquet(path: &std::path::Path, rows: usize, row_group_size: usize) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Utf8, false),
    ]));
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int32Array::from_iter_values(
            (0..rows).map(|value| value as i32),
        )),
        Arc::new(StringArray::from_iter_values(
            (0..rows).map(|value| format!("dim-{value}")),
        )),
    ];
    write_parquet(path, schema, columns, row_group_size);
}

fn write_duplicate_dim_parquet(path: &std::path::Path, rows: usize, row_group_size: usize) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Utf8, false),
    ]));
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int32Array::from_iter_values(
            (0..rows).map(|value| (value % 65_536) as i32),
        )),
        Arc::new(StringArray::from_iter_values(
            (0..rows).map(|value| format!("dup-dim-{value}")),
        )),
    ];
    write_parquet(path, schema, columns, row_group_size);
}

fn write_narrow_facts_parquet(path: &std::path::Path, rows: usize, row_group_size: usize) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("key", DataType::Int32, false),
    ]));
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int32Array::from_iter_values(
            (0..rows).map(|value| value as i32),
        )),
        Arc::new(Int32Array::from_iter_values(
            (0..rows).map(|value| (value % 65_536) as i32),
        )),
    ];
    write_parquet(path, schema, columns, row_group_size);
}

fn write_wide_pruned_facts_parquet(path: &std::path::Path, rows: usize, row_group_size: usize) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("key", DataType::Int32, false),
        Field::new("unused_i64_a", DataType::Int64, false),
        Field::new("unused_i64_b", DataType::Int64, false),
        Field::new("unused_text_a", DataType::Utf8, false),
        Field::new("unused_text_b", DataType::Utf8, false),
        Field::new("unused_text_c", DataType::Utf8, false),
        Field::new("unused_text_d", DataType::Utf8, false),
    ]));
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int32Array::from_iter_values(
            (0..rows).map(|value| value as i32),
        )),
        Arc::new(Int32Array::from_iter_values(
            (0..rows).map(|value| (value % 65_536) as i32),
        )),
        Arc::new(Int64Array::from_iter_values(
            (0..rows).map(|value| value as i64 * 11),
        )),
        Arc::new(Int64Array::from_iter_values(
            (0..rows).map(|value| value as i64 * 13),
        )),
        Arc::new(StringArray::from_iter_values(
            (0..rows).map(|value| format!("unused-a-{value}")),
        )),
        Arc::new(StringArray::from_iter_values(
            (0..rows).map(|value| format!("unused-b-{value}")),
        )),
        Arc::new(StringArray::from_iter_values(
            (0..rows).map(|value| format!("unused-c-{value}")),
        )),
        Arc::new(StringArray::from_iter_values(
            (0..rows).map(|value| format!("unused-d-{value}")),
        )),
    ];
    write_parquet(path, schema, columns, row_group_size);
}

fn write_sparse_facts_parquet(path: &std::path::Path, rows: usize, row_group_size: usize) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Utf8, false),
    ]));
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int32Array::from_iter_values(
            (0..rows).map(|value| value as i32),
        )),
        Arc::new(Int32Array::from_iter_values(
            (0..rows).map(|value| ((value % 65_536) * 17 + 3) as i32),
        )),
        Arc::new(StringArray::from_iter_values(
            (0..rows).map(|value| format!("sparse-fact-{value}")),
        )),
    ];
    write_parquet(path, schema, columns, row_group_size);
}

fn write_sparse_dim_parquet(path: &std::path::Path, rows: usize, row_group_size: usize) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Utf8, false),
    ]));
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int32Array::from_iter_values(
            (0..rows).map(|value| (value * 17 + 3) as i32),
        )),
        Arc::new(StringArray::from_iter_values(
            (0..rows).map(|value| format!("sparse-dim-{value}")),
        )),
    ];
    write_parquet(path, schema, columns, row_group_size);
}

fn write_multi_facts_parquet(path: &std::path::Path, rows: usize, row_group_size: usize) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("k1", DataType::Int32, false),
        Field::new("k2", DataType::Int32, false),
    ]));
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int32Array::from_iter_values(
            (0..rows).map(|value| value as i32),
        )),
        Arc::new(Int32Array::from_iter_values(
            (0..rows).map(|value| (value % 8192) as i32),
        )),
        Arc::new(Int32Array::from_iter_values(
            (0..rows).map(|value| ((value / 8192) % 8) as i32),
        )),
    ];
    write_parquet(path, schema, columns, row_group_size);
}

fn write_multi_dim_parquet(path: &std::path::Path, rows: usize, row_group_size: usize) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("k1", DataType::Int32, false),
        Field::new("k2", DataType::Int32, false),
        Field::new("payload", DataType::Utf8, false),
    ]));
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int32Array::from_iter_values(
            (0..rows).map(|value| (value % 8192) as i32),
        )),
        Arc::new(Int32Array::from_iter_values(
            (0..rows).map(|value| ((value / 8192) % 8) as i32),
        )),
        Arc::new(StringArray::from_iter_values(
            (0..rows).map(|value| format!("multi-dim-{value}")),
        )),
    ];
    write_parquet(path, schema, columns, row_group_size);
}

fn write_string_facts_parquet(path: &std::path::Path, rows: usize, row_group_size: usize) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("key", DataType::Utf8, false),
    ]));
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int32Array::from_iter_values(
            (0..rows).map(|value| value as i32),
        )),
        Arc::new(StringArray::from_iter_values(
            (0..rows).map(|value| format!("k{:05}", value % 32_768)),
        )),
    ];
    write_parquet(path, schema, columns, row_group_size);
}

fn write_string_dim_parquet(path: &std::path::Path, rows: usize, row_group_size: usize) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Utf8, false),
        Field::new("payload", DataType::Utf8, false),
    ]));
    let columns: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from_iter_values(
            (0..rows).map(|value| format!("k{value:05}")),
        )),
        Arc::new(StringArray::from_iter_values(
            (0..rows).map(|value| format!("string-dim-{value}")),
        )),
    ];
    write_parquet(path, schema, columns, row_group_size);
}

fn write_wide_facts_parquet(path: &std::path::Path, rows: usize, row_group_size: usize) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("key", DataType::Int32, false),
        Field::new("payload_a", DataType::Utf8, false),
        Field::new("payload_b", DataType::Utf8, false),
        Field::new("payload_c", DataType::Utf8, false),
    ]));
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int32Array::from_iter_values(
            (0..rows).map(|value| value as i32),
        )),
        Arc::new(Int32Array::from_iter_values(
            (0..rows).map(|value| (value % 32_768) as i32),
        )),
        Arc::new(StringArray::from_iter_values(
            (0..rows).map(|value| format!("wide-a-{value}")),
        )),
        Arc::new(StringArray::from_iter_values(
            (0..rows).map(|value| format!("wide-b-{value}")),
        )),
        Arc::new(StringArray::from_iter_values(
            (0..rows).map(|value| format!("wide-c-{value}")),
        )),
    ];
    write_parquet(path, schema, columns, row_group_size);
}

fn write_wide_dim_parquet(path: &std::path::Path, rows: usize, row_group_size: usize) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload_x", DataType::Utf8, false),
        Field::new("payload_y", DataType::Utf8, false),
    ]));
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int32Array::from_iter_values(
            (0..rows).map(|value| value as i32),
        )),
        Arc::new(StringArray::from_iter_values(
            (0..rows).map(|value| format!("wide-x-{value}")),
        )),
        Arc::new(StringArray::from_iter_values(
            (0..rows).map(|value| format!("wide-y-{value}")),
        )),
    ];
    write_parquet(path, schema, columns, row_group_size);
}

fn write_parquet(
    path: &std::path::Path,
    schema: Arc<Schema>,
    columns: Vec<ArrayRef>,
    row_group_size: usize,
) {
    let batch = RecordBatch::try_new(schema.clone(), columns).expect("record batch");
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_max_row_group_row_count(Some(row_group_size))
        .build();
    let file = std::fs::File::create(path).expect("create parquet");
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).expect("writer");
    writer.write(&batch).expect("write batch");
    writer.close().expect("close parquet");
}
