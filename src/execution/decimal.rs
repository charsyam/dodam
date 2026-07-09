use arrow::array::{Array, ArrayRef, Decimal128Array};
use arrow::datatypes::DataType;

use crate::error::Result;

#[derive(Clone, Copy)]
pub struct DecimalInput<'a> {
    pub values: &'a Decimal128Array,
    pub precision: u8,
    pub scale: f64,
}

impl DecimalInput<'_> {
    #[inline]
    pub fn is_null(&self, row: usize) -> bool {
        self.values.is_null(row)
    }

    #[inline]
    pub fn null_count(&self) -> usize {
        self.values.null_count()
    }

    #[inline]
    pub fn value(&self, row: usize) -> f64 {
        self.values.value(row) as f64 / self.scale
    }

    #[inline]
    pub fn raw_values(&self) -> &[i128] {
        self.values.values().as_ref()
    }

    #[inline]
    pub fn scale_i64(&self) -> Option<i64> {
        (self.scale <= i64::MAX as f64).then_some(self.scale as i64)
    }
}

pub fn decimal_input(column: &ArrayRef) -> Result<Option<DecimalInput<'_>>> {
    let DataType::Decimal128(precision, scale) = column.data_type() else {
        return Ok(None);
    };
    Ok(Some(DecimalInput {
        values: column
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .expect("Decimal128 input"),
        precision: *precision,
        scale: decimal_scale_factor(*scale),
    }))
}

#[inline]
pub fn decimal_discounted_revenue_scales(
    extendedprices: DecimalInput<'_>,
    discounts: DecimalInput<'_>,
) -> (f64, f64) {
    (
        discounts.scale,
        1.0 / (extendedprices.scale * discounts.scale),
    )
}

#[inline]
pub fn decimal_discounted_revenue_raw(
    extendedprice: i128,
    discount: i128,
    discount_scale: f64,
    revenue_scale: f64,
) -> f64 {
    (extendedprice as f64) * (discount_scale - discount as f64) * revenue_scale
}

#[inline]
pub fn decimal_discounted_revenue_raw_i64(
    extendedprice: i64,
    discount: i64,
    discount_scale: f64,
    revenue_scale: f64,
) -> f64 {
    (extendedprice as f64) * (discount_scale - discount as f64) * revenue_scale
}

#[inline]
fn decimal_scale_factor(scale: i8) -> f64 {
    10_f64.powi(i32::from(scale))
}
