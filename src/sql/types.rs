use super::*;

#[derive(Debug)]
pub enum QueryOutput {
    Scan {
        batches: Vec<RecordBatch>,
    },
    Aggregate {
        metrics: AggregateMetrics,
        batches: Vec<RecordBatch>,
    },
    Explain {
        plan: String,
    },
}

pub trait SqlResultSink {
    fn record_batch_sink(&mut self) -> &mut dyn RecordBatchSink;
    fn write_output(&mut self, output: QueryOutput) -> Result<()>;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SqlExecutionOptions {
    pub join_memory_limit_bytes: Option<u64>,
}

#[derive(Debug, Clone, Copy)]
pub struct SqlSinkExecutionOptions {
    pub allow_direct_or_streaming: bool,
}

impl Default for SqlSinkExecutionOptions {
    fn default() -> Self {
        Self {
            allow_direct_or_streaming: true,
        }
    }
}

#[derive(Debug, Default)]
pub struct SqlSinkExecutionProfile {
    pub direct_sink: Option<Duration>,
    pub streaming: Option<Duration>,
    pub execute: Option<Duration>,
    pub write_output: Option<Duration>,
    pub scan_plan_metrics: Option<ScanPlanMetrics>,
}
