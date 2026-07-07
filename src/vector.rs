use arrow::array::{
    Array, ArrayRef, Date32Array, Decimal128Array, DictionaryArray, Int64Array, LargeStringArray,
    StringArray,
};
use arrow::datatypes::Int32Type;
use arrow::record_batch::RecordBatch;

use crate::error::{DodamError, Result};

#[derive(Clone, Copy)]
pub(crate) struct BatchView<'a> {
    inner: BatchViewInner<'a>,
}

#[derive(Clone, Copy)]
enum BatchViewInner<'a> {
    RecordBatch(&'a RecordBatch),
    RawColumns(&'a [RawColumnView<'a>]),
}

#[derive(Clone, Copy)]
pub(crate) enum RawColumnView<'a> {
    I64(&'a [i64]),
    I32(&'a [i32]),
}

impl<'a> BatchView<'a> {
    pub(crate) fn new(batch: &'a RecordBatch) -> Self {
        Self {
            inner: BatchViewInner::RecordBatch(batch),
        }
    }

    pub(crate) fn from_raw_columns(columns: &'a [RawColumnView<'a>]) -> Self {
        Self {
            inner: BatchViewInner::RawColumns(columns),
        }
    }

    pub(crate) fn record_batch(&self) -> &'a RecordBatch {
        match self.inner {
            BatchViewInner::RecordBatch(batch) => batch,
            BatchViewInner::RawColumns(_) => {
                panic!("raw vector BatchView does not expose a RecordBatch")
            }
        }
    }

    pub(crate) fn num_columns(&self) -> usize {
        match self.inner {
            BatchViewInner::RecordBatch(batch) => batch.num_columns(),
            BatchViewInner::RawColumns(columns) => columns.len(),
        }
    }

    pub(crate) fn num_rows(&self) -> usize {
        match self.inner {
            BatchViewInner::RecordBatch(batch) => batch.num_rows(),
            BatchViewInner::RawColumns(columns) => {
                columns.first().map(RawColumnView::len).unwrap_or_default()
            }
        }
    }

    pub(crate) fn column(&self, index: usize) -> Result<&'a ArrayRef> {
        match self.inner {
            BatchViewInner::RecordBatch(batch) => batch.columns().get(index).ok_or_else(|| {
                DodamError::UnsupportedSql(format!("projected column index {index} missing"))
            }),
            BatchViewInner::RawColumns(_) => Err(DodamError::UnsupportedSql(format!(
                "raw vector BatchView column {index} is not an Arrow array"
            ))),
        }
    }

    pub(crate) fn raw_i64(&self, index: usize) -> Option<&'a [i64]> {
        match self.raw_column(index)? {
            RawColumnView::I64(values) => Some(values),
            _ => None,
        }
    }

    pub(crate) fn raw_i32(&self, index: usize) -> Option<&'a [i32]> {
        match self.raw_column(index)? {
            RawColumnView::I32(values) => Some(values),
            _ => None,
        }
    }

    pub(crate) fn raw_i64_i32_i32(&self) -> Option<(&'a [i64], &'a [i32], &'a [i32])> {
        Some((self.raw_i64(0)?, self.raw_i32(1)?, self.raw_i32(2)?))
    }

    fn raw_column(&self, index: usize) -> Option<RawColumnView<'a>> {
        match self.inner {
            BatchViewInner::RecordBatch(_) => None,
            BatchViewInner::RawColumns(columns) => columns.get(index).copied(),
        }
    }

    pub(crate) fn utf8(&self, index: usize) -> Option<&'a StringArray> {
        self.downcast(index)
    }

    pub(crate) fn required_utf8(&self, index: usize) -> Result<&'a StringArray> {
        self.utf8(index).ok_or_else(|| {
            DodamError::UnsupportedSql(format!("projected column {index} is not Utf8"))
        })
    }

    pub(crate) fn i64(&self, index: usize) -> Option<&'a Int64Array> {
        self.downcast(index)
    }

    pub(crate) fn required_i64(&self, index: usize) -> Result<&'a Int64Array> {
        self.i64(index).ok_or_else(|| {
            DodamError::UnsupportedSql(format!("projected column {index} is not Int64"))
        })
    }

    pub(crate) fn date32(&self, index: usize) -> Option<&'a Date32Array> {
        self.downcast(index)
    }

    pub(crate) fn required_date32(&self, index: usize) -> Result<&'a Date32Array> {
        self.date32(index).ok_or_else(|| {
            DodamError::UnsupportedSql(format!("projected column {index} is not Date32"))
        })
    }

    pub(crate) fn decimal128(&self, index: usize) -> Option<&'a Decimal128Array> {
        self.downcast(index)
    }

    pub(crate) fn dictionary_i32(&self, index: usize) -> Option<&'a DictionaryArray<Int32Type>> {
        self.downcast(index)
    }

    pub(crate) fn required_dictionary_i32(
        &self,
        index: usize,
    ) -> Result<&'a DictionaryArray<Int32Type>> {
        self.dictionary_i32(index).ok_or_else(|| {
            DodamError::UnsupportedSql(format!("projected column {index} is not Dictionary<Int32>"))
        })
    }

    fn downcast<T: 'static>(&self, index: usize) -> Option<&'a T> {
        match self.inner {
            BatchViewInner::RecordBatch(batch) => {
                batch.columns().get(index)?.as_any().downcast_ref::<T>()
            }
            BatchViewInner::RawColumns(_) => None,
        }
    }
}

impl RawColumnView<'_> {
    fn len(&self) -> usize {
        match self {
            Self::I64(values) => values.len(),
            Self::I32(values) => values.len(),
        }
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

pub(crate) enum DictionaryStringValues<'a> {
    Utf8(&'a StringArray),
    LargeUtf8(&'a LargeStringArray),
}

impl DictionaryStringValues<'_> {
    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Utf8(values) => values.len(),
            Self::LargeUtf8(values) => values.len(),
        }
    }

    pub(crate) fn value_bytes(&self, index: usize) -> &[u8] {
        match self {
            Self::Utf8(values) => {
                let offsets = values.value_offsets();
                let data = values.value_data();
                &data[offsets[index] as usize..offsets[index + 1] as usize]
            }
            Self::LargeUtf8(values) => {
                let offsets = values.value_offsets();
                let data = values.value_data();
                &data[offsets[index] as usize..offsets[index + 1] as usize]
            }
        }
    }
}

pub(crate) fn dictionary_i32_string_values(
    dictionary: &DictionaryArray<Int32Type>,
) -> Option<DictionaryStringValues<'_>> {
    if let Some(values) = dictionary.values().as_any().downcast_ref::<StringArray>() {
        return Some(DictionaryStringValues::Utf8(values));
    }
    dictionary
        .values()
        .as_any()
        .downcast_ref::<LargeStringArray>()
        .map(DictionaryStringValues::LargeUtf8)
}

pub(crate) fn dictionary_string_key_for_value(
    values: &DictionaryStringValues<'_>,
    target: &[u8],
) -> Option<i32> {
    for index in 0..values.len() {
        if values.value_bytes(index) == target {
            return i32::try_from(index).ok();
        }
    }
    None
}

pub(crate) fn dictionary_i32_match_flags(
    dictionary: &DictionaryArray<Int32Type>,
    targets: &[&[u8]],
) -> Option<Vec<Option<usize>>> {
    let values = dictionary_i32_string_values(dictionary)?;
    let mut flags = vec![None; values.len()];
    for (target_index, target) in targets.iter().enumerate() {
        let Some(key) = dictionary_string_key_for_value(&values, target) else {
            continue;
        };
        let Ok(key) = usize::try_from(key) else {
            continue;
        };
        if let Some(flag) = flags.get_mut(key) {
            *flag = Some(target_index);
        }
    }
    Some(flags)
}

#[derive(Debug, Default, Clone)]
pub(crate) struct SelectionVector {
    rows: Vec<u32>,
}

impl SelectionVector {
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self {
            rows: Vec::with_capacity(capacity),
        }
    }

    pub(crate) fn push(&mut self, row: usize) {
        self.rows.push(row as u32);
    }

    pub(crate) fn len(&self) -> usize {
        self.rows.len()
    }

    pub(crate) fn as_slice(&self) -> &[u32] {
        &self.rows
    }
}
