use std::fs::File;
use std::sync::Arc;

use arrow::array::{Int64Array, StringArray, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use dodam::engine::DodamEngine;
use dodam::sql::{QueryOutput, execute_sql};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;

#[tokio::test]
async fn q13_count_distribution_supports_keys_above_ten_million() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let customer_path = tempdir.path().join("customer.parquet");
    let orders_path = tempdir.path().join("orders.parquet");
    write_customer_parquet(&customer_path);
    write_orders_parquet(&orders_path);

    let sql = format!(
        "SELECT c_count, count(*) AS custdist
         FROM (
             SELECT c_custkey, count(o_orderkey) AS c_count
             FROM '{}' LEFT OUTER JOIN '{}'
               ON c_custkey = o_custkey
              AND o_comment NOT LIKE '%special%requests%'
             GROUP BY c_custkey
         ) AS c_orders
         GROUP BY c_count
         ORDER BY custdist DESC, c_count DESC",
        customer_path.display(),
        orders_path.display()
    );
    let output = execute_sql(&DodamEngine::default(), &sql, 2)
        .await
        .expect("execute Q13 with a high narrow key range");

    let QueryOutput::Aggregate { batches, .. } = output else {
        panic!("expected aggregate output");
    };
    assert_eq!(u64_values(&batches, 0), vec![1, 0]);
    assert_eq!(u64_values(&batches, 1), vec![2, 1]);
}

fn write_customer_parquet(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "c_custkey",
        DataType::Int64,
        false,
    )]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int64Array::from_iter_values([
            1, 15_000_000, 15_000_001,
        ]))],
    )
    .expect("customer batch");
    write_batch(path, schema, batch);
}

fn write_orders_parquet(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("o_orderkey", DataType::Int64, false),
        Field::new("o_custkey", DataType::Int64, false),
        Field::new("o_comment", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from_iter_values([10, 11, 12, 13])),
            Arc::new(Int64Array::from_iter_values([
                15_000_000, 15_000_000, 15_000_001, 1,
            ])),
            Arc::new(StringArray::from_iter_values([
                "ordinary order",
                "special pending requests",
                "regular request",
                "special requests",
            ])),
        ],
    )
    .expect("orders batch");
    write_batch(path, schema, batch);
}

fn write_batch(path: &std::path::Path, schema: Arc<Schema>, batch: RecordBatch) {
    let file = File::create(path).expect("create parquet file");
    let properties = WriterProperties::builder()
        .set_max_row_group_row_count(Some(2))
        .build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(properties)).expect("parquet writer");
    writer.write(&batch).expect("write parquet batch");
    writer.close().expect("close parquet writer");
}

fn u64_values(batches: &[RecordBatch], column: usize) -> Vec<u64> {
    batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(column)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .expect("UInt64 output")
                .values()
                .iter()
                .copied()
                .collect::<Vec<_>>()
        })
        .collect()
}
