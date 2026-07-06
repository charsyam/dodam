use arrow::array::{ArrayRef, Date32Array, DictionaryArray, Int64Array, StringArray};
use arrow::datatypes::Int32Type;
use arrow::record_batch::RecordBatch;

use crate::error::{DodamError, Result};

#[derive(Clone, Copy)]
pub(crate) struct BatchView<'a> {
    batch: &'a RecordBatch,
}

impl<'a> BatchView<'a> {
    pub(crate) fn new(batch: &'a RecordBatch) -> Self {
        Self { batch }
    }

    pub(crate) fn record_batch(&self) -> &'a RecordBatch {
        self.batch
    }

    pub(crate) fn num_columns(&self) -> usize {
        self.batch.num_columns()
    }

    pub(crate) fn num_rows(&self) -> usize {
        self.batch.num_rows()
    }

    pub(crate) fn column(&self, index: usize) -> Result<&'a ArrayRef> {
        self.batch.columns().get(index).ok_or_else(|| {
            DodamError::UnsupportedSql(format!("projected column index {index} missing"))
        })
    }

    pub(crate) fn utf8(&self, index: usize) -> Option<&'a StringArray> {
        self.downcast(index)
    }

    pub(crate) fn i64(&self, index: usize) -> Option<&'a Int64Array> {
        self.downcast(index)
    }

    pub(crate) fn date32(&self, index: usize) -> Option<&'a Date32Array> {
        self.downcast(index)
    }

    pub(crate) fn dictionary_i32(&self, index: usize) -> Option<&'a DictionaryArray<Int32Type>> {
        self.downcast(index)
    }

    fn downcast<T: 'static>(&self, index: usize) -> Option<&'a T> {
        self.batch
            .columns()
            .get(index)?
            .as_any()
            .downcast_ref::<T>()
    }
}

pub(crate) trait BatchConsumer {
    fn consume(&mut self, batch: BatchView<'_>) -> Result<()>;
}

pub(crate) fn consume_record_batch<C: BatchConsumer>(
    consumer: &mut C,
    batch: &RecordBatch,
) -> Result<()> {
    consumer.consume(BatchView::new(batch))
}

#[allow(dead_code)]
#[derive(Debug, Default, Clone)]
pub(crate) struct SelectionVector {
    rows: Vec<u32>,
}

#[allow(dead_code)]
impl SelectionVector {
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self {
            rows: Vec::with_capacity(capacity),
        }
    }

    pub(crate) fn push(&mut self, row: usize) {
        self.rows.push(row as u32);
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    pub(crate) fn len(&self) -> usize {
        self.rows.len()
    }

    pub(crate) fn as_slice(&self) -> &[u32] {
        &self.rows
    }
}
