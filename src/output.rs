use arrow::record_batch::RecordBatch;

use crate::copy::write_csv_record_batch;
use crate::error::{DodamError, Result};
use crate::execution::{AggregateMetrics, RecordBatchSink};
use crate::sql::{QueryOutput, SqlResultSink};

pub struct StdoutQuerySink;

impl StdoutQuerySink {
    pub fn new() -> Self {
        Self
    }

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
            Err(DodamError::UnsupportedSql(_)) => {
                println!("{batch:?}");
                Ok(())
            }
            Err(error) => Err(error),
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

impl Default for StdoutQuerySink {
    fn default() -> Self {
        Self::new()
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

fn nanos_to_millis(nanos: u64) -> f64 {
    nanos as f64 / 1_000_000.0
}

fn query_summary_enabled() -> bool {
    std::env::var("DODAM_QUERY_SUMMARY")
        .ok()
        .is_some_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}
