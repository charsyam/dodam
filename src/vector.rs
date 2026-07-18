use arrow::array::{
    Array, Date32Array, Decimal128Array, DictionaryArray, Int32Array, Int64Array, LargeStringArray,
    StringArray,
};
use arrow::datatypes::{DataType, Int32Type};
use arrow::record_batch::RecordBatch;
use bytes::Bytes;

use crate::dense::DenseAtomicU8;
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
    I64Bytes {
        data: &'a [u8],
        len: usize,
    },
    I64Nullable {
        values: &'a [i64],
        def_levels: &'a [i16],
    },
    I32(&'a [i32]),
    I32Bytes {
        data: &'a [u8],
        len: usize,
    },
    I32Nullable {
        values: &'a [i32],
        def_levels: &'a [i16],
    },
    Date32(&'a [i32]),
    Date32Bytes {
        data: &'a [u8],
        len: usize,
    },
    #[allow(dead_code)]
    Decimal128 {
        values: &'a [i128],
        precision: u8,
        scale: i8,
    },
    Decimal128I64 {
        values: &'a [i64],
        precision: u8,
        scale: i8,
    },
    Decimal128I64Bytes {
        data: &'a [u8],
        len: usize,
        precision: u8,
        scale: i8,
    },
    #[allow(dead_code)]
    DictionaryI32 {
        keys: &'a [i32],
        values: DictionaryStringValues<'a>,
    },
}

#[derive(Clone, Copy)]
pub(crate) enum DictionaryI32View<'a> {
    Arrow(&'a DictionaryArray<Int32Type>),
    Raw {
        keys: &'a [i32],
        values: DictionaryStringValues<'a>,
    },
}

#[derive(Clone, Copy)]
pub(crate) enum I64VectorView<'a> {
    Arrow(&'a Int64Array),
    Raw(&'a [i64]),
    RawBytes {
        data: &'a [u8],
        len: usize,
    },
    RawNullable {
        values: &'a [i64],
        def_levels: &'a [i16],
    },
}

impl<'a> I64VectorView<'a> {
    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Arrow(values) => values.len(),
            Self::Raw(values) => values.len(),
            Self::RawBytes { len, .. } => *len,
            Self::RawNullable { def_levels, .. } => def_levels.len(),
        }
    }

    pub(crate) fn is_null(&self, row: usize) -> bool {
        match self {
            Self::Arrow(values) => values.is_null(row),
            Self::Raw(_) | Self::RawBytes { .. } => false,
            Self::RawNullable { def_levels, .. } => def_levels[row] == 0,
        }
    }

    pub(crate) fn value(&self, row: usize) -> i64 {
        match self {
            Self::Arrow(values) => values.value(row),
            Self::Raw(values) => values[row],
            Self::RawBytes { data, .. } => read_i64_le_unaligned(data, row),
            Self::RawNullable { values, def_levels } => {
                if values.len() == def_levels.len() {
                    return values[row];
                }
                let value_index = nullable_value_index(def_levels, row);
                values[value_index]
            }
        }
    }

    pub(crate) fn values_if_null_free(&self) -> Option<&'a [i64]> {
        match self {
            Self::Arrow(values) => (values.null_count() == 0).then(|| values.values().as_ref()),
            Self::Raw(values) => Some(values),
            Self::RawBytes { .. } => None,
            Self::RawNullable { values, def_levels } => (values.len() == def_levels.len()
                && def_levels.iter().all(|level| *level != 0))
            .then_some(*values),
        }
    }

    pub(crate) fn raw_nullable(&self) -> Option<(&'a [i64], &'a [i16])> {
        match self {
            Self::RawNullable { values, def_levels } => Some((values, def_levels)),
            _ => None,
        }
    }

    pub(crate) fn raw_bytes(&self) -> Option<(&'a [u8], usize)> {
        match self {
            Self::RawBytes { data, len } => Some((data, *len)),
            _ => None,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum I32VectorView<'a> {
    Arrow(&'a Int32Array),
    Raw(&'a [i32]),
    RawBytes {
        data: &'a [u8],
        len: usize,
    },
    RawNullable {
        values: &'a [i32],
        def_levels: &'a [i16],
    },
}

impl<'a> I32VectorView<'a> {
    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Arrow(values) => values.len(),
            Self::Raw(values) => values.len(),
            Self::RawBytes { len, .. } => *len,
            Self::RawNullable { def_levels, .. } => def_levels.len(),
        }
    }

    pub(crate) fn is_null(&self, row: usize) -> bool {
        match self {
            Self::Arrow(values) => values.is_null(row),
            Self::Raw(_) | Self::RawBytes { .. } => false,
            Self::RawNullable { def_levels, .. } => def_levels[row] == 0,
        }
    }

    pub(crate) fn value(&self, row: usize) -> i32 {
        match self {
            Self::Arrow(values) => values.value(row),
            Self::Raw(values) => values[row],
            Self::RawBytes { data, .. } => read_i32_le_unaligned(data, row),
            Self::RawNullable { values, def_levels } => {
                if values.len() == def_levels.len() {
                    return values[row];
                }
                let value_index = nullable_value_index(def_levels, row);
                values[value_index]
            }
        }
    }

    pub(crate) fn values_if_null_free(&self) -> Option<&'a [i32]> {
        match self {
            Self::Arrow(values) => (values.null_count() == 0).then(|| values.values().as_ref()),
            Self::Raw(values) => Some(values),
            Self::RawBytes { .. } => None,
            Self::RawNullable { values, def_levels } => (values.len() == def_levels.len()
                && def_levels.iter().all(|level| *level != 0))
            .then_some(*values),
        }
    }

    pub(crate) fn raw_nullable(&self) -> Option<(&'a [i32], &'a [i16])> {
        match self {
            Self::RawNullable { values, def_levels } => Some((values, def_levels)),
            _ => None,
        }
    }

    pub(crate) fn raw_bytes(&self) -> Option<(&'a [u8], usize)> {
        match self {
            Self::RawBytes { data, len } => Some((data, *len)),
            _ => None,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum Date32VectorView<'a> {
    Arrow(&'a Date32Array),
    Raw(&'a [i32]),
    RawBytes { data: &'a [u8], len: usize },
}

impl<'a> Date32VectorView<'a> {
    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Arrow(values) => values.len(),
            Self::Raw(values) => values.len(),
            Self::RawBytes { len, .. } => *len,
        }
    }

    pub(crate) fn is_null(&self, row: usize) -> bool {
        match self {
            Self::Arrow(values) => values.is_null(row),
            Self::Raw(_) | Self::RawBytes { .. } => false,
        }
    }

    pub(crate) fn value(&self, row: usize) -> i32 {
        match self {
            Self::Arrow(values) => values.value(row),
            Self::Raw(values) => values[row],
            Self::RawBytes { data, .. } => read_i32_le_unaligned(data, row),
        }
    }

    pub(crate) fn values_if_null_free(&self) -> Option<&'a [i32]> {
        match self {
            Self::Arrow(values) => (values.null_count() == 0).then(|| values.values().as_ref()),
            Self::Raw(values) => Some(values),
            Self::RawBytes { .. } => None,
        }
    }

    pub(crate) fn raw_bytes(&self) -> Option<(&'a [u8], usize)> {
        match self {
            Self::RawBytes { data, len } => Some((data, *len)),
            _ => None,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum Utf8VectorView<'a> {
    Arrow(&'a StringArray),
}

impl<'a> Utf8VectorView<'a> {
    pub(crate) fn null_count(&self) -> usize {
        match self {
            Self::Arrow(values) => values.null_count(),
        }
    }

    pub(crate) fn is_null(&self, row: usize) -> bool {
        match self {
            Self::Arrow(values) => values.is_null(row),
        }
    }

    pub(crate) fn is_valid(&self, row: usize) -> bool {
        !self.is_null(row)
    }

    pub(crate) fn value_bytes(&self, row: usize) -> &'a [u8] {
        match self {
            Self::Arrow(values) => {
                let offsets = values.value_offsets();
                let data = values.value_data();
                &data[offsets[row] as usize..offsets[row + 1] as usize]
            }
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum Decimal128VectorView<'a> {
    Arrow {
        values: &'a Decimal128Array,
        precision: u8,
        scale: f64,
    },
    Raw {
        values: &'a [i128],
        precision: u8,
        scale: f64,
    },
    RawI64 {
        values: &'a [i64],
        precision: u8,
        scale: f64,
    },
    RawI64Bytes {
        data: &'a [u8],
        len: usize,
        precision: u8,
        scale: f64,
    },
}

impl<'a> Decimal128VectorView<'a> {
    pub(crate) fn try_new_arrow(values: &'a Decimal128Array) -> Option<Self> {
        let DataType::Decimal128(precision, scale) = values.data_type() else {
            return None;
        };
        Some(Self::Arrow {
            values,
            precision: *precision,
            scale: decimal_scale_factor(*scale),
        })
    }

    pub(crate) fn precision(&self) -> u8 {
        match self {
            Self::Arrow { precision, .. }
            | Self::Raw { precision, .. }
            | Self::RawI64 { precision, .. }
            | Self::RawI64Bytes { precision, .. } => *precision,
        }
    }

    pub(crate) fn scale(&self) -> f64 {
        match self {
            Self::Arrow { scale, .. }
            | Self::Raw { scale, .. }
            | Self::RawI64 { scale, .. }
            | Self::RawI64Bytes { scale, .. } => *scale,
        }
    }

    pub(crate) fn scale_i64(&self) -> Option<i64> {
        let scale = self.scale();
        (scale <= i64::MAX as f64).then_some(scale as i64)
    }

    pub(crate) fn null_count(&self) -> usize {
        match self {
            Self::Arrow { values, .. } => values.null_count(),
            Self::Raw { .. } | Self::RawI64 { .. } | Self::RawI64Bytes { .. } => 0,
        }
    }

    pub(crate) fn is_null(&self, row: usize) -> bool {
        match self {
            Self::Arrow { values, .. } => values.is_null(row),
            Self::Raw { .. } | Self::RawI64 { .. } | Self::RawI64Bytes { .. } => false,
        }
    }

    pub(crate) fn value(&self, row: usize) -> f64 {
        match self {
            Self::Arrow { values, scale, .. } => values.value(row) as f64 / *scale,
            Self::Raw { values, scale, .. } => values[row] as f64 / *scale,
            Self::RawI64 { values, scale, .. } => values[row] as f64 / *scale,
            Self::RawI64Bytes { data, scale, .. } => {
                read_i64_le_unaligned(data, row) as f64 / *scale
            }
        }
    }

    pub(crate) fn raw_values(&self) -> &'a [i128] {
        match self {
            Self::Arrow { values, .. } => values.values().as_ref(),
            Self::Raw { values, .. } => values,
            Self::RawI64 { .. } | Self::RawI64Bytes { .. } => {
                panic!("Decimal128VectorView::raw_values is not available for raw i64 decimals")
            }
        }
    }

    pub(crate) fn raw_i64_values(&self) -> Option<&'a [i64]> {
        match self {
            Self::RawI64 { values, .. } => Some(values),
            _ => None,
        }
    }

    pub(crate) fn raw_i64_bytes(&self) -> Option<(&'a [u8], usize)> {
        match self {
            Self::RawI64Bytes { data, len, .. } => Some((data, *len)),
            _ => None,
        }
    }

    pub(crate) fn raw_i64_value(&self, row: usize) -> Option<i64> {
        match self {
            Self::Arrow { values, .. } => Some(values.values().as_ref()[row] as i64),
            Self::Raw { values, .. } => Some(values[row] as i64),
            Self::RawI64 { values, .. } => Some(values[row]),
            Self::RawI64Bytes { data, .. } => Some(read_i64_le_unaligned(data, row)),
        }
    }

    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Arrow { values, .. } => values.len(),
            Self::Raw { values, .. } => values.len(),
            Self::RawI64 { values, .. } => values.len(),
            Self::RawI64Bytes { len, .. } => *len,
        }
    }
}

impl<'a> BatchView<'a> {
    pub(crate) fn new(batch: &'a RecordBatch) -> Self {
        Self {
            inner: BatchViewInner::RecordBatch(batch),
        }
    }

    pub(crate) fn from_raw_columns(columns: &'a [RawColumnView<'a>]) -> Self {
        debug_assert!(
            columns
                .windows(2)
                .all(|window| window[0].len() == window[1].len()),
            "raw vector BatchView columns must have equal lengths"
        );
        Self {
            inner: BatchViewInner::RawColumns(columns),
        }
    }

    pub(crate) fn try_record_batch(&self) -> Option<&'a RecordBatch> {
        match self.inner {
            BatchViewInner::RecordBatch(batch) => Some(batch),
            BatchViewInner::RawColumns(_) => None,
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

    pub(crate) fn raw_i64(&self, index: usize) -> Option<&'a [i64]> {
        match self.raw_column(index)? {
            RawColumnView::I64(values) => Some(values),
            _ => None,
        }
    }

    #[allow(dead_code)]
    pub(crate) fn raw_i32(&self, index: usize) -> Option<&'a [i32]> {
        match self.raw_column(index)? {
            RawColumnView::I32(values) => Some(values),
            _ => None,
        }
    }

    pub(crate) fn raw_i64_i32_i32(&self) -> Option<(&'a [i64], &'a [i32], &'a [i32])> {
        Some((
            self.raw_i64(0)?,
            self.raw_i32_physical(1)?,
            self.raw_i32_physical(2)?,
        ))
    }

    fn raw_column(&self, index: usize) -> Option<RawColumnView<'a>> {
        match self.inner {
            BatchViewInner::RecordBatch(_) => None,
            BatchViewInner::RawColumns(columns) => columns.get(index).copied(),
        }
    }

    fn raw_i32_physical(&self, index: usize) -> Option<&'a [i32]> {
        match self.raw_column(index)? {
            RawColumnView::I32(values) | RawColumnView::Date32(values) => Some(values),
            _ => None,
        }
    }

    pub(crate) fn utf8(&self, index: usize) -> Option<&'a StringArray> {
        self.downcast(index)
    }

    pub(crate) fn utf8_vector(&self, index: usize) -> Option<Utf8VectorView<'a>> {
        match self.inner {
            BatchViewInner::RecordBatch(_) => self.utf8(index).map(Utf8VectorView::Arrow),
            BatchViewInner::RawColumns(_) => None,
        }
    }

    pub(crate) fn required_utf8(&self, index: usize) -> Result<&'a StringArray> {
        self.utf8(index).ok_or_else(|| {
            DodamError::UnsupportedSql(format!("projected column {index} is not Utf8"))
        })
    }

    pub(crate) fn i64(&self, index: usize) -> Option<&'a Int64Array> {
        self.downcast(index)
    }

    pub(crate) fn i64_vector(&self, index: usize) -> Option<I64VectorView<'a>> {
        match self.inner {
            BatchViewInner::RecordBatch(_) => self.i64(index).map(I64VectorView::Arrow),
            BatchViewInner::RawColumns(_) => match self.raw_column(index)? {
                RawColumnView::I64(values) => Some(I64VectorView::Raw(values)),
                RawColumnView::I64Bytes { data, len } => {
                    Some(I64VectorView::RawBytes { data, len })
                }
                RawColumnView::I64Nullable { values, def_levels } => {
                    Some(I64VectorView::RawNullable { values, def_levels })
                }
                _ => None,
            },
        }
    }

    pub(crate) fn required_i64(&self, index: usize) -> Result<&'a Int64Array> {
        self.i64(index).ok_or_else(|| {
            DodamError::UnsupportedSql(format!("projected column {index} is not Int64"))
        })
    }

    pub(crate) fn i32(&self, index: usize) -> Option<&'a Int32Array> {
        self.downcast(index)
    }

    pub(crate) fn i32_vector(&self, index: usize) -> Option<I32VectorView<'a>> {
        match self.inner {
            BatchViewInner::RecordBatch(_) => self.i32(index).map(I32VectorView::Arrow),
            BatchViewInner::RawColumns(_) => match self.raw_column(index)? {
                RawColumnView::I32(values) => Some(I32VectorView::Raw(values)),
                RawColumnView::I32Bytes { data, len } => {
                    Some(I32VectorView::RawBytes { data, len })
                }
                RawColumnView::I32Nullable { values, def_levels } => {
                    Some(I32VectorView::RawNullable { values, def_levels })
                }
                _ => None,
            },
        }
    }

    pub(crate) fn date32(&self, index: usize) -> Option<&'a Date32Array> {
        self.downcast(index)
    }

    pub(crate) fn date32_vector(&self, index: usize) -> Option<Date32VectorView<'a>> {
        match self.inner {
            BatchViewInner::RecordBatch(_) => self.date32(index).map(Date32VectorView::Arrow),
            BatchViewInner::RawColumns(_) => match self.raw_column(index)? {
                RawColumnView::Date32(values) => Some(Date32VectorView::Raw(values)),
                RawColumnView::Date32Bytes { data, len } => {
                    Some(Date32VectorView::RawBytes { data, len })
                }
                _ => None,
            },
        }
    }

    pub(crate) fn decimal128(&self, index: usize) -> Option<&'a Decimal128Array> {
        self.downcast(index)
    }

    pub(crate) fn decimal128_vector(&self, index: usize) -> Option<Decimal128VectorView<'a>> {
        match self.inner {
            BatchViewInner::RecordBatch(_) => {
                Decimal128VectorView::try_new_arrow(self.decimal128(index)?)
            }
            BatchViewInner::RawColumns(_) => match self.raw_column(index)? {
                RawColumnView::Decimal128 {
                    values,
                    precision,
                    scale,
                } => Some(Decimal128VectorView::Raw {
                    values,
                    precision,
                    scale: decimal_scale_factor(scale),
                }),
                RawColumnView::Decimal128I64 {
                    values,
                    precision,
                    scale,
                } => Some(Decimal128VectorView::RawI64 {
                    values,
                    precision,
                    scale: decimal_scale_factor(scale),
                }),
                RawColumnView::Decimal128I64Bytes {
                    data,
                    len,
                    precision,
                    scale,
                } => Some(Decimal128VectorView::RawI64Bytes {
                    data,
                    len,
                    precision,
                    scale: decimal_scale_factor(scale),
                }),
                _ => None,
            },
        }
    }

    pub(crate) fn dictionary_i32(&self, index: usize) -> Option<&'a DictionaryArray<Int32Type>> {
        self.downcast(index)
    }

    pub(crate) fn dictionary_i32_view(&self, index: usize) -> Option<DictionaryI32View<'a>> {
        match self.inner {
            BatchViewInner::RecordBatch(_) => {
                self.dictionary_i32(index).map(DictionaryI32View::Arrow)
            }
            BatchViewInner::RawColumns(_) => match self.raw_column(index)? {
                RawColumnView::DictionaryI32 { keys, values } => {
                    Some(DictionaryI32View::Raw { keys, values })
                }
                _ => None,
            },
        }
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
            Self::I64Bytes { len, .. } => *len,
            Self::I64Nullable { def_levels, .. } => def_levels.len(),
            Self::I32(values) | Self::Date32(values) => values.len(),
            Self::I32Bytes { len, .. } | Self::Date32Bytes { len, .. } => *len,
            Self::I32Nullable { def_levels, .. } => def_levels.len(),
            Self::Decimal128 { values, .. } => values.len(),
            Self::Decimal128I64 { values, .. } => values.len(),
            Self::Decimal128I64Bytes { len, .. } => *len,
            Self::DictionaryI32 { keys, .. } => keys.len(),
        }
    }
}

#[inline]
pub(crate) fn read_i32_le_unaligned(data: &[u8], row: usize) -> i32 {
    let offset = row * std::mem::size_of::<i32>();
    let mut bytes = [0u8; 4];
    bytes.copy_from_slice(&data[offset..offset + 4]);
    i32::from_le_bytes(bytes)
}

#[inline]
pub(crate) fn read_i64_le_unaligned(data: &[u8], row: usize) -> i64 {
    let offset = row * std::mem::size_of::<i64>();
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&data[offset..offset + 8]);
    i64::from_le_bytes(bytes)
}

fn nullable_value_index(def_levels: &[i16], row: usize) -> usize {
    def_levels[..row]
        .iter()
        .filter(|level| **level != 0)
        .count()
}

#[inline]
fn decimal_scale_factor(scale: i8) -> f64 {
    10_f64.powi(i32::from(scale))
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

#[derive(Clone, Copy)]
pub(crate) enum DictionaryStringValues<'a> {
    Utf8(&'a StringArray),
    LargeUtf8(&'a LargeStringArray),
    #[allow(dead_code)]
    Bytes(&'a [Bytes]),
}

impl<'a> DictionaryStringValues<'a> {
    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Utf8(values) => values.len(),
            Self::LargeUtf8(values) => values.len(),
            Self::Bytes(values) => values.len(),
        }
    }

    pub(crate) fn value_bytes(&self, index: usize) -> &'a [u8] {
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
            Self::Bytes(values) => values[index].as_ref(),
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

pub(crate) fn dictionary_i32_view_match_flags(
    dictionary: DictionaryI32View<'_>,
    targets: &[&[u8]],
) -> Option<Vec<Option<usize>>> {
    let values = dictionary.string_values()?;
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

pub(crate) fn dictionary_i32_view_match_index(
    dictionary_keys: &[i32],
    match_flags: &[Option<usize>],
    row: usize,
) -> Option<usize> {
    let key = dictionary_keys
        .get(row)
        .copied()
        .and_then(|key| usize::try_from(key).ok())?;
    match_flags.get(key).copied().flatten()
}

pub(crate) fn store_i64_keys_matching_dictionary_target(
    keys: I64VectorView<'_>,
    dictionary: DictionaryI32View<'_>,
    target: &[u8],
    markers: &DenseAtomicU8,
) -> bool {
    let Some(match_flags) = dictionary_i32_view_match_flags(dictionary, &[target]) else {
        return false;
    };
    let dictionary_keys = dictionary.keys();
    if dictionary.null_count() == 0
        && let Some(key_values) = keys.values_if_null_free()
    {
        for (row, key) in key_values.iter().copied().enumerate() {
            if dictionary_i32_row_matches(dictionary_keys, &match_flags, row)
                && let Ok(index) = usize::try_from(key)
            {
                markers.store_present(index);
            }
        }
        return true;
    }
    for row in 0..keys.len() {
        if keys.is_null(row)
            || dictionary.is_null(row)
            || !dictionary_i32_row_matches(dictionary_keys, &match_flags, row)
        {
            continue;
        }
        if let Ok(index) = usize::try_from(keys.value(row)) {
            markers.store_present(index);
        }
    }
    true
}

pub(crate) fn store_i64_keys_matching_utf8_target(
    keys: I64VectorView<'_>,
    strings: &StringArray,
    target: &[u8],
    markers: &DenseAtomicU8,
) {
    if strings.null_count() == 0
        && let Some(key_values) = keys.values_if_null_free()
    {
        let offsets = strings.value_offsets();
        let data = strings.value_data();
        for (row, key) in key_values.iter().copied().enumerate() {
            if string_array_value_bytes(offsets, data, row) == target
                && let Ok(index) = usize::try_from(key)
            {
                markers.store_present(index);
            }
        }
        return;
    }
    for row in 0..keys.len() {
        if keys.is_null(row) || strings.is_null(row) || strings.value(row).as_bytes() != target {
            continue;
        }
        if let Ok(index) = usize::try_from(keys.value(row)) {
            markers.store_present(index);
        }
    }
}

fn dictionary_i32_row_matches(
    dictionary_keys: &[i32],
    match_flags: &[Option<usize>],
    row: usize,
) -> bool {
    dictionary_i32_view_match_index(dictionary_keys, match_flags, row).is_some()
}

fn string_array_value_bytes<'a>(offsets: &[i32], data: &'a [u8], row: usize) -> &'a [u8] {
    let start = offsets[row] as usize;
    let end = offsets[row + 1] as usize;
    &data[start..end]
}

impl<'a> DictionaryI32View<'a> {
    pub(crate) fn keys(&self) -> &[i32] {
        match self {
            Self::Arrow(dictionary) => dictionary.keys().values().as_ref(),
            Self::Raw { keys, .. } => keys,
        }
    }

    pub(crate) fn null_count(&self) -> usize {
        match self {
            Self::Arrow(dictionary) => dictionary.null_count(),
            Self::Raw { .. } => 0,
        }
    }

    pub(crate) fn is_null(&self, index: usize) -> bool {
        match self {
            Self::Arrow(dictionary) => dictionary.is_null(index),
            Self::Raw { .. } => false,
        }
    }

    pub(crate) fn string_values(&self) -> Option<DictionaryStringValues<'_>> {
        self.string_values_for_view()
    }

    pub(crate) fn string_values_for_view(&self) -> Option<DictionaryStringValues<'a>> {
        match self {
            Self::Arrow(dictionary) => dictionary_i32_string_values(dictionary),
            Self::Raw { values, .. } => Some(*values),
        }
    }
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

#[derive(Debug, Clone, Copy)]
pub(crate) struct SelectionRuns<'a> {
    runs: &'a [(usize, usize)],
    selected_rows: usize,
}

impl<'a> SelectionRuns<'a> {
    pub(crate) fn new(runs: &'a [(usize, usize)], selected_rows: usize) -> Self {
        Self {
            runs,
            selected_rows,
        }
    }

    pub(crate) fn runs(&self) -> &'a [(usize, usize)] {
        self.runs
    }

    pub(crate) fn selected_rows(&self) -> usize {
        self.selected_rows
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.selected_rows == 0
    }
}

#[derive(Debug, Default)]
pub(crate) struct SelectionRunsBuilder {
    selected_rows: usize,
}

impl SelectionRunsBuilder {
    pub(crate) fn selected_rows(&self) -> usize {
        self.selected_rows
    }

    pub(crate) fn push_run(&mut self, runs: &mut Vec<(usize, usize)>, start: usize, len: usize) {
        if len == 0 {
            return;
        }
        self.selected_rows = self.selected_rows.saturating_add(len);
        if let Some((previous_start, previous_len)) = runs.last_mut()
            && previous_start.saturating_add(*previous_len) == start
        {
            *previous_len = previous_len.saturating_add(len);
            return;
        }
        runs.push((start, len));
    }

    pub(crate) fn push_disjoint_run(
        &mut self,
        runs: &mut Vec<(usize, usize)>,
        start: usize,
        len: usize,
    ) {
        if len == 0 {
            return;
        }
        self.selected_rows = self.selected_rows.saturating_add(len);
        runs.push((start, len));
    }
}
