use super::*;

pub(super) fn apply_output_distinct(
    batches: Vec<RecordBatch>,
    distinct: bool,
) -> Result<Vec<RecordBatch>> {
    if !distinct {
        return Ok(batches);
    }
    collect_batches(Box::new(DistinctExec::new(Box::new(MemoryExec::new(batches)))).execute()?)
}

pub(super) fn limit_batches(
    batches: Vec<RecordBatch>,
    limit: Option<usize>,
    offset: usize,
) -> Vec<RecordBatch> {
    let mut to_skip = offset;
    let mut remaining = limit.unwrap_or(usize::MAX);
    let mut limited = Vec::new();
    for batch in batches {
        if to_skip >= batch.num_rows() {
            to_skip -= batch.num_rows();
            continue;
        }
        let batch = if to_skip > 0 {
            let sliced = batch.slice(to_skip, batch.num_rows() - to_skip);
            to_skip = 0;
            sliced
        } else {
            batch
        };
        if remaining == 0 {
            break;
        }
        let rows = remaining.min(batch.num_rows());
        remaining -= rows;
        if rows > 0 {
            limited.push(batch.slice(0, rows));
        }
    }
    limited
}
