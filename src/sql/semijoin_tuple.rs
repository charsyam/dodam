use super::*;
use crate::sql::semijoin::semijoin_i64_key_at;

pub(super) enum TupleSemijoinPairSet {
    I64(SemijoinI64PairKeys),
    I64Utf8(SemijoinI64Utf8PairKeys),
    Literal(SemijoinLiteralPairKeys),
}

impl TupleSemijoinPairSet {
    pub(super) fn has_null(&self) -> bool {
        match self {
            Self::I64(keys) => keys.has_null,
            Self::I64Utf8(keys) => keys.has_null,
            Self::Literal(keys) => keys.has_null,
        }
    }
}

pub(super) struct SemijoinI64Utf8PairKeys {
    values: SemijoinI64Utf8PairValues,
    numeric_values: SemijoinNumericValues,
    string_ids: FastHashMap<Vec<u8>, u32>,
    pub(super) numeric_left: bool,
    has_null: bool,
}

pub(super) struct SemijoinMixedPrefixCandidateKeys {
    numeric_values: SemijoinNumericValues,
    string_ids: FastHashMap<Vec<u8>, u32>,
    numeric_left: bool,
}

pub(super) enum SemijoinNumericValues {
    I64(FastHashSet<i64>),
    I32(FastHashSet<i32>),
    I32Dense {
        min: i32,
        contains: Vec<u8>,
        len: usize,
    },
}

impl SemijoinNumericValues {
    fn empty_i64() -> Self {
        Self::I64(FastHashSet::default())
    }

    fn empty_i32() -> Self {
        Self::I32(FastHashSet::default())
    }

    fn reserve(&mut self, additional: usize) {
        match self {
            Self::I64(values) => values.reserve(additional),
            Self::I32(values) => values.reserve(additional),
            Self::I32Dense { .. } => {}
        }
    }

    fn insert_i32(&mut self, value: i32) {
        match self {
            Self::I32(values) => {
                values.insert(value);
            }
            Self::I64(values) => {
                values.insert(i64::from(value));
            }
            Self::I32Dense { .. } => {}
        }
    }

    fn insert_i64(&mut self, value: i64) {
        match self {
            Self::I64(values) => {
                values.insert(value);
            }
            Self::I32(values) => {
                if let Ok(value) = i32::try_from(value) {
                    values.insert(value);
                } else {
                    let mut wide = FastHashSet::<i64>::with_capacity_and_hasher(
                        values.len().saturating_add(1),
                        Default::default(),
                    );
                    wide.extend(values.drain().map(i64::from));
                    wide.insert(value);
                    *self = Self::I64(wide);
                }
            }
            Self::I32Dense { .. } => {}
        }
    }

    fn contains_i32(&self, value: i32) -> bool {
        match self {
            Self::I64(values) => values.contains(&i64::from(value)),
            Self::I32(values) => values.contains(&value),
            Self::I32Dense { min, contains, .. } => value
                .checked_sub(*min)
                .and_then(|offset| usize::try_from(offset).ok())
                .is_some_and(|index| contains.get(index).is_some_and(|byte| *byte != 0)),
        }
    }

    fn contains_i64(&self, value: i64) -> bool {
        match self {
            Self::I64(values) => values.contains(&value),
            Self::I32(values) => i32::try_from(value).is_ok_and(|value| values.contains(&value)),
            Self::I32Dense { .. } => {
                i32::try_from(value).is_ok_and(|value| self.contains_i32(value))
            }
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::I64(values) => values.len(),
            Self::I32(values) => values.len(),
            Self::I32Dense { len, .. } => *len,
        }
    }

    fn kind(&self) -> &'static str {
        match self {
            Self::I64(_) => "i64-hash",
            Self::I32(_) => "i32-hash",
            Self::I32Dense { .. } => "i32-dense",
        }
    }

    fn optimize_dense_i32(self) -> Self {
        let Self::I32(values) = self else {
            return self;
        };
        let len = values.len();
        if len == 0 {
            return Self::I32(values);
        }
        let Some(min) = values.iter().copied().min() else {
            return Self::I32(values);
        };
        let Some(max) = values.iter().copied().max() else {
            return Self::I32(values);
        };
        let span = i64::from(max) - i64::from(min) + 1;
        if span <= 0 {
            return Self::I32(values);
        }
        let Ok(span) = usize::try_from(span) else {
            return Self::I32(values);
        };
        let max_span = len.saturating_mul(mixed_tuple_dense_numeric_max_span_factor());
        if span > max_span || span > mixed_tuple_dense_numeric_max_bytes() {
            return Self::I32(values);
        }
        let mut contains = vec![0u8; span];
        for value in values {
            let offset = i64::from(value) - i64::from(min);
            if let Ok(index) = usize::try_from(offset)
                && let Some(slot) = contains.get_mut(index)
            {
                *slot = 1;
            }
        }
        Self::I32Dense { min, contains, len }
    }
}

pub(super) fn mixed_tuple_dense_numeric_max_span_factor() -> usize {
    std::env::var("DODAM_MIXED_TUPLE_DENSE_NUMERIC_MAX_SPAN_FACTOR")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(8)
}

pub(super) fn mixed_tuple_dense_numeric_max_bytes() -> usize {
    std::env::var("DODAM_MIXED_TUPLE_DENSE_NUMERIC_MAX_BYTES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(16 * 1024 * 1024)
}

pub(super) enum SemijoinI64Utf8PairValues {
    Pair(FastHashSet<(i64, u32)>),
    PackedI32U32(FastHashSet<u64>),
    DenseI32ToU32 {
        min: i32,
        values: Vec<u32>,
        present: Vec<u8>,
        len: usize,
    },
}

impl SemijoinI64Utf8PairValues {
    fn empty_packed() -> Self {
        Self::PackedI32U32(FastHashSet::default())
    }

    fn reserve(&mut self, additional: usize) {
        match self {
            Self::Pair(values) => values.reserve(additional),
            Self::PackedI32U32(values) => values.reserve(additional),
            Self::DenseI32ToU32 { .. } => {}
        }
    }

    fn insert(&mut self, number: i64, string_id: u32) {
        match self {
            Self::PackedI32U32(values) => {
                if let Some(key) = pack_i32_u32_pair(number, string_id) {
                    values.insert(key);
                } else {
                    let mut pair_values = FastHashSet::with_capacity_and_hasher(
                        values.len().saturating_add(1),
                        Default::default(),
                    );
                    for key in values.drain() {
                        let (number, string_id) = unpack_i32_u32_pair(key);
                        pair_values.insert((number, string_id));
                    }
                    pair_values.insert((number, string_id));
                    *self = Self::Pair(pair_values);
                }
            }
            Self::Pair(values) => {
                values.insert((number, string_id));
            }
            Self::DenseI32ToU32 { .. } => {
                let mut values = FastHashSet::with_capacity_and_hasher(1, Default::default());
                values.insert((number, string_id));
                *self = Self::Pair(values);
            }
        }
    }

    fn contains(&self, number: i64, string_id: u32) -> bool {
        match self {
            Self::PackedI32U32(values) => {
                pack_i32_u32_pair(number, string_id).is_some_and(|key| values.contains(&key))
            }
            Self::Pair(values) => values.contains(&(number, string_id)),
            Self::DenseI32ToU32 {
                min,
                values,
                present,
                ..
            } => {
                let Ok(number) = i32::try_from(number) else {
                    return false;
                };
                number
                    .checked_sub(*min)
                    .and_then(|offset| usize::try_from(offset).ok())
                    .is_some_and(|index| {
                        present.get(index).is_some_and(|byte| *byte != 0)
                            && values.get(index).is_some_and(|value| *value == string_id)
                    })
            }
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Pair(values) => values.len(),
            Self::PackedI32U32(values) => values.len(),
            Self::DenseI32ToU32 { len, .. } => *len,
        }
    }

    fn kind(&self) -> &'static str {
        match self {
            Self::Pair(_) => "i64-u32-hash",
            Self::PackedI32U32(_) => "i32-u32-hash",
            Self::DenseI32ToU32 { .. } => "i32-u32-dense",
        }
    }

    fn is_dense_i32_to_u32(&self) -> bool {
        matches!(self, Self::DenseI32ToU32 { .. })
    }
}

pub(super) fn mixed_tuple_dense_pair_max_span_factor() -> usize {
    std::env::var("DODAM_MIXED_TUPLE_DENSE_PAIR_MAX_SPAN_FACTOR")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(8)
}

pub(super) fn mixed_tuple_dense_pair_max_bytes() -> usize {
    std::env::var("DODAM_MIXED_TUPLE_DENSE_PAIR_MAX_BYTES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(16 * 1024 * 1024)
}

pub(super) fn semijoin_i32_u32_pairs_to_values(
    pairs: Vec<(i32, u32)>,
) -> SemijoinI64Utf8PairValues {
    if std::env::var_os("DODAM_DISABLE_MIXED_TUPLE_DENSE_PAIR_MAP").is_none()
        && let Some(values) = semijoin_i32_u32_pairs_to_dense_values(&pairs)
    {
        return values;
    }
    let mut values = FastHashSet::with_capacity_and_hasher(pairs.len(), Default::default());
    for (number, string_id) in pairs {
        if let Some(key) = pack_i32_u32_pair(i64::from(number), string_id) {
            values.insert(key);
        }
    }
    SemijoinI64Utf8PairValues::PackedI32U32(values)
}

pub(super) fn semijoin_i32_u32_pairs_to_dense_values(
    pairs: &[(i32, u32)],
) -> Option<SemijoinI64Utf8PairValues> {
    if pairs.is_empty() {
        return Some(SemijoinI64Utf8PairValues::DenseI32ToU32 {
            min: 0,
            values: Vec::new(),
            present: Vec::new(),
            len: 0,
        });
    }
    let min = pairs.iter().map(|(number, _)| *number).min()?;
    let max = pairs.iter().map(|(number, _)| *number).max()?;
    let span = i64::from(max) - i64::from(min) + 1;
    if span <= 0 {
        return None;
    }
    let span = usize::try_from(span).ok()?;
    let max_span = pairs
        .len()
        .saturating_mul(mixed_tuple_dense_pair_max_span_factor());
    if span > max_span || span > mixed_tuple_dense_pair_max_bytes() {
        return None;
    }
    let mut values = vec![0u32; span];
    let mut present = vec![0u8; span];
    let mut len = 0usize;
    for &(number, string_id) in pairs {
        let index = usize::try_from(i64::from(number) - i64::from(min)).ok()?;
        if present[index] != 0 {
            if values[index] != string_id {
                return None;
            }
        } else {
            present[index] = 1;
            values[index] = string_id;
            len += 1;
        }
    }
    Some(SemijoinI64Utf8PairValues::DenseI32ToU32 {
        min,
        values,
        present,
        len,
    })
}

pub(super) fn pack_i32_u32_pair(number: i64, string_id: u32) -> Option<u64> {
    let number = u32::try_from(i32::try_from(number).ok()?).ok()?;
    Some((u64::from(number) << 32) | u64::from(string_id))
}

pub(super) fn unpack_i32_u32_pair(key: u64) -> (i64, u32) {
    let number = (key >> 32) as u32;
    (
        i64::from(i32::from_ne_bytes(number.to_ne_bytes())),
        key as u32,
    )
}

pub(super) struct SemijoinLiteralPairKeys {
    values: FastHashSet<(String, String)>,
    has_null: bool,
}

pub(super) fn tuple_semijoin_pair_mask(
    batch: &RecordBatch,
    left_column: &str,
    right_column: &str,
    keys: &TupleSemijoinPairSet,
    negated: bool,
) -> Result<BooleanArray> {
    match keys {
        TupleSemijoinPairSet::I64(keys) => {
            semijoin_i64_pair_mask(batch, left_column, right_column, keys, negated)
        }
        TupleSemijoinPairSet::I64Utf8(keys) => {
            semijoin_i64_utf8_pair_mask(batch, left_column, right_column, keys, negated)
        }
        TupleSemijoinPairSet::Literal(keys) => {
            semijoin_literal_pair_mask(batch, left_column, right_column, keys, negated)
        }
    }
}

pub(super) async fn collect_semijoin_i64_pair_set(
    engine: &DodamEngine,
    path: PathBuf,
    left_column: &str,
    right_column: &str,
    filter: Option<FilterExpr>,
    batch_size: usize,
) -> Result<SemijoinI64PairKeys> {
    if std::env::var_os("DODAM_DISABLE_I32_PAIR_DIRECT_BUILD").is_none()
        && let Some(keys) = collect_semijoin_i32_pair_set_direct(
            engine,
            &path,
            left_column,
            right_column,
            filter.as_ref(),
            batch_size,
        )?
    {
        return Ok(keys);
    }
    let mut projection = vec![left_column.to_string(), right_column.to_string()];
    if let Some(filter) = &filter {
        for column in filter.referenced_columns() {
            add_column_once(&mut projection, column);
        }
    }
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(projection),
            filter.clone(),
        )
        .await?;
    let mut values = None::<SemijoinI64PairSet>;
    let mut has_null = false;
    for batch in stream.by_ref() {
        let batch = batch?;
        let left_index = batch_column_index(&batch, left_column)?;
        let right_index = batch_column_index(&batch, right_column)?;
        let left = batch.column(left_index);
        let right = batch.column(right_index);
        has_null |= left.null_count() > 0 || right.null_count() > 0;
        let values = values.get_or_insert_with(|| SemijoinI64PairSet::for_arrays(left, right));
        values.insert_from_arrays(left, right)?;
    }
    Ok(SemijoinI64PairKeys {
        values: values.unwrap_or_else(SemijoinI64PairSet::empty_pair),
        has_null,
    })
}

pub(super) fn collect_semijoin_i32_pair_set_direct(
    engine: &DodamEngine,
    path: &Path,
    left_column: &str,
    right_column: &str,
    filter: Option<&FilterExpr>,
    batch_size: usize,
) -> Result<Option<SemijoinI64PairKeys>> {
    let Some(filter_expr) = filter.map(FilterExpr::expr) else {
        return Ok(None);
    };
    if !semijoin_i32_pair_filter_supported(filter_expr, left_column, right_column) {
        return Ok(None);
    }
    let row_groups = (0..engine.parquet_row_group_count(path)?).collect::<Vec<_>>();
    let mut pair_values = Vec::<(i32, i32)>::new();
    let mut has_null = false;
    let metrics = engine.scan_parquet_i32_i32_columns(
        path,
        batch_size,
        &row_groups,
        [left_column, right_column],
        |left_values, left_def_levels, right_values, right_def_levels| {
            pair_values.reserve(left_values.len());
            for row in 0..left_values.len() {
                let left = (!left_def_levels.is_some_and(|levels| levels[row] == 0))
                    .then_some(left_values[row]);
                let right = (!right_def_levels.is_some_and(|levels| levels[row] == 0))
                    .then_some(right_values[row]);
                if !semijoin_i32_pair_nullable_filter_matches(
                    filter_expr,
                    left_column,
                    right_column,
                    left,
                    right,
                ) {
                    continue;
                }
                match (left, right) {
                    (Some(left), Some(right)) => {
                        pair_values.push((left, right));
                    }
                    _ => {
                        has_null = true;
                    }
                }
            }
            Ok(Some(()))
        },
    )?;
    if metrics.is_none() {
        return Ok(None);
    }
    let values = if std::env::var_os("DODAM_DISABLE_I32_PAIR_DENSE_MAP").is_none() {
        semijoin_i32_pair_values_to_dense_map(&pair_values).unwrap_or_else(|| {
            SemijoinI64PairSet::PackedI32(
                pair_values
                    .iter()
                    .map(|(left, right)| pack_i32_pair(*left, *right))
                    .collect(),
            )
        })
    } else {
        SemijoinI64PairSet::PackedI32(
            pair_values
                .iter()
                .map(|(left, right)| pack_i32_pair(*left, *right))
                .collect(),
        )
    };
    Ok(Some(SemijoinI64PairKeys { values, has_null }))
}

pub(super) fn semijoin_i32_pair_values_to_dense_map(
    pairs: &[(i32, i32)],
) -> Option<SemijoinI64PairSet> {
    if pairs.is_empty() {
        return Some(SemijoinI64PairSet::DenseI32ToI32 {
            min: 0,
            values: Vec::new(),
            present: Vec::new(),
        });
    }
    let min = pairs.iter().map(|(left, _)| *left).min()?;
    let max = pairs.iter().map(|(left, _)| *left).max()?;
    let span = i64::from(max) - i64::from(min) + 1;
    if span <= 0 {
        return None;
    }
    let span = usize::try_from(span).ok()?;
    let max_span = pairs
        .len()
        .saturating_mul(i32_pair_dense_map_max_span_factor());
    if span > max_span || span > i32_pair_dense_map_max_bytes() {
        return None;
    }
    let mut values = vec![0i32; span];
    let mut present = vec![0u8; span];
    for &(left, right) in pairs {
        let index = usize::try_from(i64::from(left) - i64::from(min)).ok()?;
        if present[index] != 0 {
            if values[index] != right {
                return None;
            }
        } else {
            present[index] = 1;
            values[index] = right;
        }
    }
    Some(SemijoinI64PairSet::DenseI32ToI32 {
        min,
        values,
        present,
    })
}

pub(super) fn i32_pair_dense_map_max_span_factor() -> usize {
    std::env::var("DODAM_I32_PAIR_DENSE_MAP_MAX_SPAN_FACTOR")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(4)
}

pub(super) fn i32_pair_dense_map_max_bytes() -> usize {
    std::env::var("DODAM_I32_PAIR_DENSE_MAP_MAX_BYTES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(16 * 1024 * 1024)
}

pub(super) fn semijoin_i32_pair_filter_supported(
    expr: &Expr,
    left_column: &str,
    right_column: &str,
) -> bool {
    match expr {
        Expr::Boolean(_) => true,
        Expr::And(left, right) => {
            semijoin_i32_pair_filter_supported(left, left_column, right_column)
                && semijoin_i32_pair_filter_supported(right, left_column, right_column)
        }
        Expr::Comparison(ComparisonExpr { column, value, .. }) => {
            (column == left_column || column == right_column)
                && matches!(value, LiteralValue::Int64(value) if i32::try_from(*value).is_ok())
        }
        Expr::IsNull { column, .. } => column == left_column || column == right_column,
        _ => false,
    }
}

pub(super) fn semijoin_i32_pair_nullable_filter_matches(
    expr: &Expr,
    left_column: &str,
    right_column: &str,
    left: Option<i32>,
    right: Option<i32>,
) -> bool {
    match expr {
        Expr::Boolean(value) => value.unwrap_or(false),
        Expr::And(lhs, rhs) => {
            semijoin_i32_pair_nullable_filter_matches(lhs, left_column, right_column, left, right)
                && semijoin_i32_pair_nullable_filter_matches(
                    rhs,
                    left_column,
                    right_column,
                    left,
                    right,
                )
        }
        Expr::Comparison(ComparisonExpr { column, op, value }) => {
            let LiteralValue::Int64(value) = value else {
                return false;
            };
            let Ok(value) = i32::try_from(*value) else {
                return false;
            };
            let input = if column == left_column {
                left
            } else if column == right_column {
                right
            } else {
                return false;
            };
            input.is_some_and(|input| compare_i32(input, *op, value))
        }
        Expr::IsNull { column, negated } => {
            let input = if column == left_column {
                left
            } else if column == right_column {
                right
            } else {
                return false;
            };
            let is_null = input.is_none();
            if *negated { !is_null } else { is_null }
        }
        _ => false,
    }
}

pub(super) async fn collect_semijoin_literal_pair_set(
    engine: &DodamEngine,
    path: PathBuf,
    left_column: &str,
    right_column: &str,
    filter: Option<FilterExpr>,
    batch_size: usize,
) -> Result<SemijoinLiteralPairKeys> {
    let mut projection = vec![left_column.to_string(), right_column.to_string()];
    if let Some(filter) = &filter {
        for column in filter.referenced_columns() {
            add_column_once(&mut projection, column);
        }
    }
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(projection),
            filter.clone(),
        )
        .await?;
    let mut values = FastHashSet::<(String, String)>::default();
    let mut has_null = false;
    for batch in stream.by_ref() {
        let batch = batch?;
        let left_index = batch_column_index(&batch, left_column)?;
        let right_index = batch_column_index(&batch, right_column)?;
        let left = batch.column(left_index);
        let right = batch.column(right_index);
        has_null |= left.null_count() > 0 || right.null_count() > 0;
        for row in 0..batch.num_rows() {
            let Some(left) = semijoin_key_at(left, row)? else {
                continue;
            };
            let Some(right) = semijoin_key_at(right, row)? else {
                continue;
            };
            values.insert((left, right));
        }
    }
    Ok(SemijoinLiteralPairKeys { values, has_null })
}

pub(super) async fn collect_semijoin_i64_utf8_pair_set(
    engine: &DodamEngine,
    path: PathBuf,
    left_column: &str,
    right_column: &str,
    filter: Option<FilterExpr>,
    batch_size: usize,
) -> Result<Option<SemijoinI64Utf8PairKeys>> {
    if std::env::var_os("DODAM_DISABLE_MIXED_TUPLE_DIRECT_BUILD").is_none()
        && let Some(keys) = collect_semijoin_i64_utf8_pair_set_direct(
            engine,
            &path,
            left_column,
            right_column,
            filter.as_ref(),
            batch_size,
        )?
    {
        return Ok(Some(keys));
    }
    let mut projection = vec![left_column.to_string(), right_column.to_string()];
    if let Some(filter) = &filter {
        for column in filter.referenced_columns() {
            add_column_once(&mut projection, column);
        }
    }
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(projection),
            filter.clone(),
        )
        .await?;
    let mut values = SemijoinI64Utf8PairValues::empty_packed();
    let mut numeric_values = SemijoinNumericValues::empty_i64();
    let mut string_ids = FastHashMap::<Vec<u8>, u32>::default();
    let mut numeric_left = None::<bool>;
    let mut has_null = false;
    for batch in stream.by_ref() {
        let batch = batch?;
        let left_index = batch_column_index(&batch, left_column)?;
        let right_index = batch_column_index(&batch, right_column)?;
        let left = batch.column(left_index);
        let right = batch.column(right_index);
        has_null |= left.null_count() > 0 || right.null_count() > 0;
        let batch_numeric_left = if semijoin_i64_array(left).is_some()
            && (right.as_any().downcast_ref::<StringArray>().is_some()
                || semijoin_dictionary_i32_view(right).is_some())
        {
            true
        } else if (left.as_any().downcast_ref::<StringArray>().is_some()
            || semijoin_dictionary_i32_view(left).is_some())
            && semijoin_i64_array(right).is_some()
        {
            false
        } else {
            return Ok(None);
        };
        if let Some(existing) = numeric_left {
            if existing != batch_numeric_left {
                return Ok(None);
            }
        } else {
            numeric_left = Some(batch_numeric_left);
        }
        insert_semijoin_i64_utf8_pairs(
            &mut values,
            &mut numeric_values,
            &mut string_ids,
            left,
            right,
            batch_numeric_left,
        )?;
    }
    Ok(numeric_left.map(|numeric_left| SemijoinI64Utf8PairKeys {
        values,
        numeric_values,
        string_ids,
        numeric_left,
        has_null,
    }))
}

pub(super) fn collect_semijoin_i64_utf8_pair_set_direct(
    engine: &DodamEngine,
    path: &Path,
    left_column: &str,
    right_column: &str,
    filter: Option<&FilterExpr>,
    batch_size: usize,
) -> Result<Option<SemijoinI64Utf8PairKeys>> {
    let Some((filter_column, prefix)) = semijoin_like_prefix_filter(filter) else {
        if semijoin_profile_enabled() {
            eprintln!("[dodam:semijoin-profile] mixed-direct-build skip=unsupported-filter");
        }
        return Ok(None);
    };
    let row_groups = (0..engine.parquet_row_group_count(path)?).collect::<Vec<_>>();
    if row_groups.is_empty() {
        return Ok(Some(SemijoinI64Utf8PairKeys {
            values: SemijoinI64Utf8PairValues::empty_packed(),
            numeric_values: SemijoinNumericValues::empty_i32(),
            string_ids: FastHashMap::default(),
            numeric_left: true,
            has_null: false,
        }));
    }
    if filter_column == right_column {
        if semijoin_profile_enabled() {
            eprintln!(
                "[dodam:semijoin-profile] mixed-direct-build attempt numeric={left_column} string={right_column}"
            );
        }
        return collect_semijoin_i32_utf8_pair_set_direct(
            engine,
            path,
            batch_size,
            &row_groups,
            left_column,
            right_column,
            prefix,
            true,
        );
    }
    if filter_column == left_column {
        if semijoin_profile_enabled() {
            eprintln!(
                "[dodam:semijoin-profile] mixed-direct-build attempt numeric={right_column} string={left_column}"
            );
        }
        return collect_semijoin_i32_utf8_pair_set_direct(
            engine,
            path,
            batch_size,
            &row_groups,
            right_column,
            left_column,
            prefix,
            false,
        );
    }
    if semijoin_profile_enabled() {
        eprintln!(
            "[dodam:semijoin-profile] mixed-direct-build skip=filter-column-not-tuple-key filter={filter_column}"
        );
    }
    Ok(None)
}

pub(super) fn collect_semijoin_mixed_prefix_candidate_keys_direct(
    engine: &DodamEngine,
    path: &Path,
    left_column: &str,
    right_column: &str,
    filter: Option<&FilterExpr>,
    batch_size: usize,
) -> Result<Option<SemijoinMixedPrefixCandidateKeys>> {
    let Some((filter_column, prefix)) = semijoin_like_prefix_filter(filter) else {
        return Ok(None);
    };
    let row_groups = (0..engine.parquet_row_group_count(path)?).collect::<Vec<_>>();
    if row_groups.is_empty() {
        return Ok(Some(SemijoinMixedPrefixCandidateKeys {
            numeric_values: SemijoinNumericValues::empty_i32(),
            string_ids: FastHashMap::default(),
            numeric_left: true,
        }));
    }
    if filter_column == right_column {
        collect_semijoin_i32_utf8_prefix_candidate_keys_direct(
            engine,
            path,
            batch_size,
            &row_groups,
            left_column,
            right_column,
            prefix,
            true,
        )
    } else if filter_column == left_column {
        collect_semijoin_i32_utf8_prefix_candidate_keys_direct(
            engine,
            path,
            batch_size,
            &row_groups,
            right_column,
            left_column,
            prefix,
            false,
        )
    } else {
        Ok(None)
    }
}

pub(super) fn collect_semijoin_i32_utf8_prefix_candidate_keys_direct(
    engine: &DodamEngine,
    path: &Path,
    batch_size: usize,
    row_groups: &[usize],
    numeric_column: &str,
    string_column: &str,
    prefix: &str,
    numeric_left: bool,
) -> Result<Option<SemijoinMixedPrefixCandidateKeys>> {
    let mut numeric_values = SemijoinNumericValues::empty_i32();
    let mut string_ids = FastHashMap::<Vec<u8>, u32>::default();
    let mut string_id_cache = SemijoinDictionaryStringIdCache::default();
    let metrics = engine.scan_parquet_i32_dictionary_id_columns(
        path,
        batch_size,
        row_groups,
        [numeric_column, string_column],
        |numbers, dictionary_def_levels, dictionary_ids, dictionary| {
            let (selected_string_ids, selected_count) = string_id_cache.refresh_prefix_dense(
                dictionary,
                &mut string_ids,
                prefix.as_bytes(),
            )?;
            if selected_count == 0 {
                return Ok(Some(()));
            }
            numeric_values.reserve(numbers.len());
            let mut dictionary_value_offset = 0usize;
            match dictionary_def_levels {
                Some(levels) => {
                    for (row, level) in levels.iter().copied().enumerate() {
                        if level == 0 {
                            continue;
                        }
                        let Some(dictionary_id) = dictionary_ids.get(dictionary_value_offset)
                        else {
                            return Ok(None);
                        };
                        dictionary_value_offset += 1;
                        if *dictionary_id >= 0
                            && selected_string_ids
                                .get(*dictionary_id as usize)
                                .copied()
                                .is_some_and(|id| id != u32::MAX)
                        {
                            numeric_values.insert_i32(numbers[row]);
                        }
                    }
                }
                None => {
                    for (row, dictionary_id) in dictionary_ids.iter().copied().enumerate() {
                        if dictionary_id >= 0
                            && selected_string_ids
                                .get(dictionary_id as usize)
                                .copied()
                                .is_some_and(|id| id != u32::MAX)
                        {
                            numeric_values.insert_i32(numbers[row]);
                        }
                    }
                    dictionary_value_offset = dictionary_ids.len();
                }
            }
            if dictionary_value_offset != dictionary_ids.len() {
                return Ok(None);
            }
            Ok(Some(()))
        },
    )?;
    if metrics.is_none() {
        return Ok(None);
    }
    numeric_values = numeric_values.optimize_dense_i32();
    if semijoin_profile_enabled() {
        eprintln!(
            "[dodam:semijoin-profile] mixed-prefix-precheck rhs numeric_values={} strings={} numeric={}",
            numeric_values.len(),
            string_ids.len(),
            numeric_values.kind()
        );
    }
    Ok(Some(SemijoinMixedPrefixCandidateKeys {
        numeric_values,
        string_ids,
        numeric_left,
    }))
}

pub(super) fn collect_semijoin_i32_utf8_pair_set_direct(
    engine: &DodamEngine,
    path: &Path,
    batch_size: usize,
    row_groups: &[usize],
    numeric_column: &str,
    string_column: &str,
    prefix: &str,
    numeric_left: bool,
) -> Result<Option<SemijoinI64Utf8PairKeys>> {
    let mut pair_values = Vec::<(i32, u32)>::new();
    let mut numeric_values = SemijoinNumericValues::empty_i32();
    let mut string_ids = FastHashMap::<Vec<u8>, u32>::default();
    let mut has_null = false;
    if std::env::var_os("DODAM_MIXED_TUPLE_SELECTED_INNER_BUILD").is_some()
        && let Some(keys) = collect_semijoin_i32_utf8_pair_set_direct_selected_inner(
            engine,
            path,
            row_groups,
            numeric_column,
            string_column,
            prefix,
            numeric_left,
        )?
    {
        return Ok(Some(keys));
    }
    if std::env::var_os("DODAM_DISABLE_MIXED_TUPLE_PARALLEL_RAW_INNER").is_none()
        && let Some(keys) = collect_semijoin_i32_utf8_pair_set_direct_parallel_raw(
            engine,
            path,
            batch_size,
            row_groups,
            numeric_column,
            string_column,
            prefix,
            numeric_left,
        )?
    {
        return Ok(Some(keys));
    }
    let mut string_id_cache = SemijoinDictionaryStringIdCache::default();
    let raw_metrics = engine.scan_parquet_i32_dictionary_id_columns_raw(
        path,
        batch_size,
        row_groups,
        [numeric_column, string_column],
        |number_bytes, records, dictionary_def_levels, dictionary_ids, dictionary| {
            let (selected_string_ids, selected_count) = string_id_cache.refresh_prefix_dense(
                dictionary,
                &mut string_ids,
                prefix.as_bytes(),
            )?;
            if selected_count == 0 {
                if let Some(levels) = dictionary_def_levels {
                    has_null |= levels.iter().any(|level| *level == 0);
                }
                return Ok(Some(()));
            }
            pair_values.reserve(records);
            numeric_values.reserve(records);
            let mut dictionary_value_offset = 0usize;
            match dictionary_def_levels {
                Some(levels) => {
                    has_null |= levels.iter().any(|level| *level == 0);
                    for (row, level) in levels.iter().copied().enumerate() {
                        if level == 0 {
                            continue;
                        }
                        let Some(dictionary_id) = dictionary_ids.get(dictionary_value_offset)
                        else {
                            return Ok(None);
                        };
                        dictionary_value_offset += 1;
                        insert_direct_semijoin_i32_utf8_pair_value_dense(
                            read_i32_le_unchecked(number_bytes, row),
                            *dictionary_id,
                            selected_string_ids,
                            &mut pair_values,
                            &mut numeric_values,
                        );
                    }
                }
                None => {
                    insert_direct_semijoin_i32_utf8_pair_batch_dense_from_bytes(
                        number_bytes,
                        records,
                        dictionary_ids,
                        selected_string_ids,
                        &mut pair_values,
                        &mut numeric_values,
                    );
                    dictionary_value_offset = dictionary_ids.len();
                }
            }
            if dictionary_value_offset != dictionary_ids.len() {
                return Ok(None);
            }
            Ok(Some(()))
        },
    )?;
    let metrics =
        if raw_metrics.is_some() {
            raw_metrics
        } else {
            pair_values.clear();
            numeric_values = SemijoinNumericValues::empty_i32();
            string_ids.clear();
            has_null = false;
            string_id_cache = SemijoinDictionaryStringIdCache::default();
            engine.scan_parquet_i32_dictionary_id_columns(
                path,
                batch_size,
                row_groups,
                [numeric_column, string_column],
                |numbers, dictionary_def_levels, dictionary_ids, dictionary| {
                    let (selected_string_ids, selected_count) = string_id_cache
                        .refresh_prefix_dense(dictionary, &mut string_ids, prefix.as_bytes())?;
                    if selected_count == 0 {
                        if let Some(levels) = dictionary_def_levels {
                            has_null |= levels.iter().any(|level| *level == 0);
                        }
                        return Ok(Some(()));
                    }
                    pair_values.reserve(numbers.len());
                    numeric_values.reserve(numbers.len());
                    let mut dictionary_value_offset = 0usize;
                    match dictionary_def_levels {
                        Some(levels) => {
                            has_null |= levels.iter().any(|level| *level == 0);
                            for (row, level) in levels.iter().copied().enumerate() {
                                if level == 0 {
                                    continue;
                                }
                                let Some(dictionary_id) =
                                    dictionary_ids.get(dictionary_value_offset)
                                else {
                                    return Ok(None);
                                };
                                dictionary_value_offset += 1;
                                insert_direct_semijoin_i32_utf8_pair_value_dense(
                                    numbers[row],
                                    *dictionary_id,
                                    selected_string_ids,
                                    &mut pair_values,
                                    &mut numeric_values,
                                );
                            }
                        }
                        None => {
                            insert_direct_semijoin_i32_utf8_pair_batch_dense(
                                numbers,
                                dictionary_ids,
                                selected_string_ids,
                                &mut pair_values,
                                &mut numeric_values,
                            );
                            dictionary_value_offset = dictionary_ids.len();
                        }
                    }
                    if dictionary_value_offset != dictionary_ids.len() {
                        return Ok(None);
                    }
                    Ok(Some(()))
                },
            )?
        };
    if metrics.is_none() {
        if semijoin_profile_enabled() {
            eprintln!(
                "[dodam:semijoin-profile] mixed-direct-build dictionary-id unsupported; trying bytearray"
            );
        }
        return collect_semijoin_i32_utf8_pair_set_direct_byte_array(
            engine,
            path,
            batch_size,
            row_groups,
            numeric_column,
            string_column,
            prefix,
            numeric_left,
        );
    }
    numeric_values = numeric_values.optimize_dense_i32();
    let values = semijoin_i32_u32_pairs_to_values(pair_values);
    if semijoin_profile_enabled() {
        eprintln!(
            "[dodam:semijoin-profile] mixed-direct-build dictionary-id ok values={} pair={} strings={} numeric={}",
            values.len(),
            values.kind(),
            string_ids.len(),
            numeric_values.kind()
        );
    }
    Ok(Some(SemijoinI64Utf8PairKeys {
        values,
        numeric_values,
        string_ids,
        numeric_left,
        has_null,
    }))
}

pub(super) struct SemijoinI32Utf8PairPartial {
    pair_values: Vec<(i32, u32)>,
    has_null: bool,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn collect_semijoin_i32_utf8_pair_set_direct_parallel_raw(
    engine: &DodamEngine,
    path: &Path,
    batch_size: usize,
    row_groups: &[usize],
    numeric_column: &str,
    string_column: &str,
    prefix: &str,
    numeric_left: bool,
) -> Result<Option<SemijoinI64Utf8PairKeys>> {
    if row_groups.len() <= 1 {
        return Ok(None);
    }
    let workers = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(4)
        .min(row_groups.len());
    if workers <= 1 {
        return Ok(None);
    }
    let partitions = semijoin_partition_row_groups(row_groups, workers);
    let shared_string_ids = Arc::new(std::sync::Mutex::new(FastHashMap::<Vec<u8>, u32>::default()));
    let (sender, receiver) = mpsc::channel();
    std::thread::scope(|scope| {
        for partition in partitions {
            let sender = sender.clone();
            let engine = engine.clone();
            let path = path.to_path_buf();
            let numeric_column = numeric_column.to_string();
            let string_column = string_column.to_string();
            let prefix = prefix.as_bytes().to_vec();
            let shared_string_ids = shared_string_ids.clone();
            scope.spawn(move || {
                let result: Result<Option<SemijoinI32Utf8PairPartial>> = (|| {
                    let mut partial = SemijoinI32Utf8PairPartial {
                        pair_values: Vec::new(),
                        has_null: false,
                    };
                    let mut string_id_cache = SemijoinDictionaryStringIdCache::default();
                    let metrics = engine.scan_parquet_i32_dictionary_id_columns_raw(
                        &path,
                        batch_size,
                        &partition,
                        [&numeric_column, &string_column],
                        |number_bytes,
                         records,
                         dictionary_def_levels,
                         dictionary_ids,
                         dictionary| {
                            let mut string_ids = shared_string_ids.lock().map_err(|_| {
                                DodamError::UnsupportedSql(
                                    "mixed tuple string id mutex poisoned".to_string(),
                                )
                            })?;
                            let (selected_string_ids, selected_count) = string_id_cache
                                .refresh_prefix_dense(dictionary, &mut string_ids, &prefix)?;
                            drop(string_ids);
                            if selected_count == 0 {
                                if let Some(levels) = dictionary_def_levels {
                                    partial.has_null |= levels.iter().any(|level| *level == 0);
                                }
                                return Ok(Some(()));
                            }
                            partial.pair_values.reserve(records);
                            let mut dictionary_value_offset = 0usize;
                            match dictionary_def_levels {
                                Some(levels) => {
                                    partial.has_null |= levels.iter().any(|level| *level == 0);
                                    for (row, level) in levels.iter().copied().enumerate() {
                                        if level == 0 {
                                            continue;
                                        }
                                        let Some(dictionary_id) =
                                            dictionary_ids.get(dictionary_value_offset)
                                        else {
                                            return Ok(None);
                                        };
                                        dictionary_value_offset += 1;
                                        insert_direct_semijoin_i32_utf8_pair_value_dense_no_numeric(
                                            read_i32_le_unchecked(number_bytes, row),
                                            *dictionary_id,
                                            selected_string_ids,
                                            &mut partial.pair_values,
                                        );
                                    }
                                }
                                None => {
                                    insert_direct_semijoin_i32_utf8_pair_batch_dense_from_bytes_no_numeric(
                                        number_bytes,
                                        records,
                                        dictionary_ids,
                                        selected_string_ids,
                                        &mut partial.pair_values,
                                    );
                                    dictionary_value_offset = dictionary_ids.len();
                                }
                            }
                            if dictionary_value_offset != dictionary_ids.len() {
                                return Ok(None);
                            }
                            Ok(Some(()))
                        },
                    )?;
                    Ok(metrics.map(|_| partial))
                })();
                let _ = sender.send(result);
            });
        }
        drop(sender);
        let mut pair_values = Vec::<(i32, u32)>::new();
        let mut numeric_values = SemijoinNumericValues::empty_i32();
        let mut has_null = false;
        for result in receiver {
            let Some(partial) = result? else {
                return Ok::<Option<SemijoinI64Utf8PairKeys>, DodamError>(None);
            };
            has_null |= partial.has_null;
            pair_values.reserve(partial.pair_values.len());
            numeric_values.reserve(partial.pair_values.len());
            for (number, string_id) in partial.pair_values {
                numeric_values.insert_i32(number);
                pair_values.push((number, string_id));
            }
        }
        let string_ids = Arc::try_unwrap(shared_string_ids)
            .map_err(|_| {
                DodamError::UnsupportedSql(
                    "mixed tuple string id map still shared after parallel build".to_string(),
                )
            })?
            .into_inner()
            .map_err(|_| {
                DodamError::UnsupportedSql("mixed tuple string id mutex poisoned".to_string())
            })?;
        numeric_values = numeric_values.optimize_dense_i32();
        let values = semijoin_i32_u32_pairs_to_values(pair_values);
        if semijoin_profile_enabled() {
            eprintln!(
                "[dodam:semijoin-profile] mixed-direct-build parallel-raw ok values={} pair={} strings={} numeric={}",
                values.len(),
                values.kind(),
                string_ids.len(),
                numeric_values.kind()
            );
        }
        Ok(Some(SemijoinI64Utf8PairKeys {
            values,
            numeric_values,
            string_ids,
            numeric_left,
            has_null,
        }))
    })
}

pub(super) fn semijoin_partition_row_groups(
    row_groups: &[usize],
    partitions: usize,
) -> Vec<Vec<usize>> {
    let partitions = partitions.min(row_groups.len()).max(1);
    let chunk_size = row_groups.len().div_ceil(partitions).max(1);
    row_groups
        .chunks(chunk_size)
        .map(|chunk| chunk.to_vec())
        .collect()
}

pub(super) fn collect_semijoin_i32_utf8_pair_set_direct_selected_inner(
    engine: &DodamEngine,
    path: &Path,
    row_groups: &[usize],
    numeric_column: &str,
    string_column: &str,
    prefix: &str,
    numeric_left: bool,
) -> Result<Option<SemijoinI64Utf8PairKeys>> {
    let mut pair_values = Vec::<(i32, u32)>::new();
    let mut numeric_values = SemijoinNumericValues::empty_i32();
    let mut string_ids = FastHashMap::<Vec<u8>, u32>::default();
    let mut string_id_cache = SemijoinDictionaryStringIdCache::default();
    let metrics = engine.scan_parquet_i32_selected_by_byte_array_prefix(
        path,
        row_groups,
        [numeric_column, string_column],
        prefix.as_bytes(),
        |numbers, dictionary_ids, dictionary| {
            let (selected_string_ids, selected_count) = string_id_cache.refresh_prefix_dense(
                dictionary,
                &mut string_ids,
                prefix.as_bytes(),
            )?;
            if selected_count == 0 {
                return Ok(Some(()));
            }
            if numbers.len() != dictionary_ids.len() {
                return Ok(None);
            }
            pair_values.reserve(numbers.len());
            numeric_values.reserve(numbers.len());
            insert_direct_semijoin_i32_utf8_pair_batch_dense(
                numbers,
                dictionary_ids,
                selected_string_ids,
                &mut pair_values,
                &mut numeric_values,
            );
            Ok(Some(()))
        },
    )?;
    if metrics.is_none() {
        return Ok(None);
    }
    numeric_values = numeric_values.optimize_dense_i32();
    let values = semijoin_i32_u32_pairs_to_values(pair_values);
    if semijoin_profile_enabled() {
        eprintln!(
            "[dodam:semijoin-profile] mixed-direct-build selected-inner ok values={} pair={} strings={} numeric={}",
            values.len(),
            values.kind(),
            string_ids.len(),
            numeric_values.kind()
        );
    }
    Ok(Some(SemijoinI64Utf8PairKeys {
        values,
        numeric_values,
        string_ids,
        numeric_left,
        has_null: false,
    }))
}

pub(super) fn collect_semijoin_i32_utf8_pair_set_direct_byte_array(
    engine: &DodamEngine,
    path: &Path,
    batch_size: usize,
    row_groups: &[usize],
    numeric_column: &str,
    string_column: &str,
    prefix: &str,
    numeric_left: bool,
) -> Result<Option<SemijoinI64Utf8PairKeys>> {
    let mut pair_values = Vec::<(i32, u32)>::new();
    let mut numeric_values = SemijoinNumericValues::empty_i32();
    let mut string_ids = FastHashMap::<Vec<u8>, u32>::default();
    let mut has_null = false;
    let metrics = engine.scan_parquet_i32_byte_array_columns(
        path,
        batch_size,
        row_groups,
        [numeric_column, string_column],
        |numbers, string_def_levels, strings| {
            pair_values.reserve(numbers.len());
            numeric_values.reserve(numbers.len());
            let mut string_value_offset = 0usize;
            for (row, level) in string_def_levels.iter().copied().enumerate() {
                if level == 0 {
                    has_null = true;
                    continue;
                }
                let Some(value) = strings.get(string_value_offset) else {
                    return Ok(None);
                };
                string_value_offset += 1;
                if !value.as_ref().starts_with(prefix.as_bytes()) {
                    continue;
                }
                let string_id = semijoin_intern_string_id(&mut string_ids, value.as_ref())?;
                numeric_values.insert_i32(numbers[row]);
                pair_values.push((numbers[row], string_id));
            }
            if string_value_offset != strings.len() {
                return Ok(None);
            }
            Ok(Some(()))
        },
    )?;
    if metrics.is_none() {
        if semijoin_profile_enabled() {
            eprintln!("[dodam:semijoin-profile] mixed-direct-build bytearray unsupported");
        }
        return Ok(None);
    }
    numeric_values = numeric_values.optimize_dense_i32();
    let values = semijoin_i32_u32_pairs_to_values(pair_values);
    if semijoin_profile_enabled() {
        eprintln!(
            "[dodam:semijoin-profile] mixed-direct-build bytearray ok values={} pair={} strings={} numeric={}",
            values.len(),
            values.kind(),
            string_ids.len(),
            numeric_values.kind()
        );
    }
    Ok(Some(SemijoinI64Utf8PairKeys {
        values,
        numeric_values,
        string_ids,
        numeric_left,
        has_null,
    }))
}

#[derive(Default)]
pub(super) struct SemijoinDictionaryStringIdCache {
    ptr: *const bytes::Bytes,
    len: usize,
    fingerprint: u64,
    selected_count: usize,
    ids: Vec<Option<u32>>,
    dense_ids: Vec<u32>,
}

impl SemijoinDictionaryStringIdCache {
    fn refresh_prefix<'a>(
        &'a mut self,
        dictionary: &[bytes::Bytes],
        string_ids: &mut FastHashMap<Vec<u8>, u32>,
        prefix: &[u8],
    ) -> Result<&'a [Option<u32>]> {
        let fingerprint = semijoin_dictionary_fingerprint(dictionary);
        if self.ptr == dictionary.as_ptr()
            && self.len == dictionary.len()
            && self.fingerprint == fingerprint
        {
            return Ok(&self.ids);
        }
        self.ptr = dictionary.as_ptr();
        self.len = dictionary.len();
        self.fingerprint = fingerprint;
        self.selected_count = 0;
        self.ids.clear();
        self.dense_ids.clear();
        self.ids.reserve(dictionary.len());
        self.dense_ids.reserve(dictionary.len());
        for value in dictionary {
            let id = if value.as_ref().starts_with(prefix) {
                semijoin_intern_string_id(string_ids, value.as_ref())?
            } else {
                u32::MAX
            };
            if id != u32::MAX {
                self.selected_count += 1;
            }
            self.ids.push((id != u32::MAX).then_some(id));
            self.dense_ids.push(id);
        }
        Ok(&self.ids)
    }

    fn refresh_prefix_dense<'a>(
        &'a mut self,
        dictionary: &[bytes::Bytes],
        string_ids: &mut FastHashMap<Vec<u8>, u32>,
        prefix: &[u8],
    ) -> Result<(&'a [u32], usize)> {
        let _ = self.refresh_prefix(dictionary, string_ids, prefix)?;
        Ok((&self.dense_ids, self.selected_count))
    }

    fn refresh_existing<'a>(
        &'a mut self,
        dictionary: &[bytes::Bytes],
        string_ids: &FastHashMap<Vec<u8>, u32>,
    ) -> &'a [Option<u32>] {
        let fingerprint = semijoin_dictionary_fingerprint(dictionary);
        if self.ptr == dictionary.as_ptr()
            && self.len == dictionary.len()
            && self.fingerprint == fingerprint
        {
            return &self.ids;
        }
        self.ptr = dictionary.as_ptr();
        self.len = dictionary.len();
        self.fingerprint = fingerprint;
        self.selected_count = 0;
        self.ids.clear();
        self.dense_ids.clear();
        self.ids.reserve(dictionary.len());
        self.dense_ids.reserve(dictionary.len());
        for value in dictionary {
            let id = string_ids.get(value.as_ref()).copied();
            if id.is_some() {
                self.selected_count += 1;
            }
            self.ids.push(id);
            self.dense_ids.push(id.unwrap_or(u32::MAX));
        }
        &self.ids
    }

    fn refresh_existing_dense<'a>(
        &'a mut self,
        dictionary: &[bytes::Bytes],
        string_ids: &FastHashMap<Vec<u8>, u32>,
    ) -> &'a [u32] {
        let _ = self.refresh_existing(dictionary, string_ids);
        &self.dense_ids
    }
}

pub(super) fn semijoin_dictionary_fingerprint(dictionary: &[bytes::Bytes]) -> u64 {
    let mut hash = dictionary.len() as u64;
    for value in dictionary {
        hash = hash.wrapping_mul(0x9E37_79B1_85EB_CA87);
        hash ^= value.len() as u64;
        for byte in value.as_ref() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x100_0000_01B3);
        }
    }
    hash
}

pub(super) fn insert_direct_semijoin_i32_utf8_pair_value_dense(
    number: i32,
    dictionary_id: i32,
    selected_string_ids: &[u32],
    pair_values: &mut Vec<(i32, u32)>,
    numeric_values: &mut SemijoinNumericValues,
) {
    let Ok(dictionary_id) = usize::try_from(dictionary_id) else {
        return;
    };
    let Some(&string_id) = selected_string_ids.get(dictionary_id) else {
        return;
    };
    if string_id == u32::MAX {
        return;
    }
    numeric_values.insert_i32(number);
    pair_values.push((number, string_id));
}

pub(super) fn insert_direct_semijoin_i32_utf8_pair_value_dense_no_numeric(
    number: i32,
    dictionary_id: i32,
    selected_string_ids: &[u32],
    pair_values: &mut Vec<(i32, u32)>,
) {
    let Ok(dictionary_id) = usize::try_from(dictionary_id) else {
        return;
    };
    let Some(&string_id) = selected_string_ids.get(dictionary_id) else {
        return;
    };
    if string_id == u32::MAX {
        return;
    }
    pair_values.push((number, string_id));
}

pub(super) fn insert_direct_semijoin_i32_utf8_pair_batch_dense(
    numbers: &[i32],
    dictionary_ids: &[i32],
    selected_string_ids: &[u32],
    pair_values: &mut Vec<(i32, u32)>,
    numeric_values: &mut SemijoinNumericValues,
) {
    let rows = numbers.len().min(dictionary_ids.len());
    for row in 0..rows {
        let dictionary_id = dictionary_ids[row];
        if dictionary_id < 0 {
            continue;
        }
        let dictionary_id = dictionary_id as usize;
        if dictionary_id >= selected_string_ids.len() {
            continue;
        }
        let string_id = selected_string_ids[dictionary_id];
        if string_id == u32::MAX {
            continue;
        }
        let number = numbers[row];
        numeric_values.insert_i32(number);
        pair_values.push((number, string_id));
    }
}

pub(super) fn insert_direct_semijoin_i32_utf8_pair_batch_dense_from_bytes(
    number_bytes: &[u8],
    records: usize,
    dictionary_ids: &[i32],
    selected_string_ids: &[u32],
    pair_values: &mut Vec<(i32, u32)>,
    numeric_values: &mut SemijoinNumericValues,
) {
    let rows = records.min(dictionary_ids.len());
    for row in 0..rows {
        let dictionary_id = dictionary_ids[row];
        if dictionary_id < 0 {
            continue;
        }
        let dictionary_id = dictionary_id as usize;
        if dictionary_id >= selected_string_ids.len() {
            continue;
        }
        let string_id = selected_string_ids[dictionary_id];
        if string_id == u32::MAX {
            continue;
        }
        let number = read_i32_le_unchecked(number_bytes, row);
        numeric_values.insert_i32(number);
        pair_values.push((number, string_id));
    }
}

pub(super) fn insert_direct_semijoin_i32_utf8_pair_batch_dense_from_bytes_no_numeric(
    number_bytes: &[u8],
    records: usize,
    dictionary_ids: &[i32],
    selected_string_ids: &[u32],
    pair_values: &mut Vec<(i32, u32)>,
) {
    let rows = records.min(dictionary_ids.len());
    for row in 0..rows {
        let dictionary_id = dictionary_ids[row];
        if dictionary_id < 0 {
            continue;
        }
        let dictionary_id = dictionary_id as usize;
        if dictionary_id >= selected_string_ids.len() {
            continue;
        }
        let string_id = selected_string_ids[dictionary_id];
        if string_id == u32::MAX {
            continue;
        }
        pair_values.push((read_i32_le_unchecked(number_bytes, row), string_id));
    }
}

pub(super) fn semijoin_like_prefix_filter(filter: Option<&FilterExpr>) -> Option<(&str, &str)> {
    let Expr::Like {
        column,
        pattern,
        negated: false,
        escape: None,
        case_insensitive: false,
    } = filter?.expr()
    else {
        return None;
    };
    let prefix = pattern.strip_suffix('%')?;
    if prefix.is_empty()
        || !prefix.is_ascii()
        || prefix
            .as_bytes()
            .iter()
            .any(|byte| matches!(byte, b'%' | b'_'))
    {
        return None;
    }
    Some((column.as_str(), prefix))
}

pub(super) fn semijoin_filter_prefers_mixed_tuple(
    filter: Option<&FilterExpr>,
    left_column: &str,
    right_column: &str,
) -> bool {
    semijoin_like_prefix_filter(filter)
        .map(|(column, _)| column == left_column || column == right_column)
        .unwrap_or(false)
}

pub(super) enum SemijoinI64Array<'a> {
    I32(&'a Int32Array),
    I64(&'a Int64Array),
}

pub(super) fn semijoin_i64_array(array: &ArrayRef) -> Option<SemijoinI64Array<'_>> {
    if let Some(array) = array.as_any().downcast_ref::<Int32Array>() {
        Some(SemijoinI64Array::I32(array))
    } else {
        array
            .as_any()
            .downcast_ref::<Int64Array>()
            .map(SemijoinI64Array::I64)
    }
}

pub(super) fn semijoin_i64_array_value(array: &SemijoinI64Array<'_>, row: usize) -> i64 {
    match array {
        SemijoinI64Array::I32(array) => i64::from(array.value(row)),
        SemijoinI64Array::I64(array) => array.value(row),
    }
}

pub(super) fn semijoin_i64_array_is_valid(array: &SemijoinI64Array<'_>, row: usize) -> bool {
    match array {
        SemijoinI64Array::I32(array) => array.is_valid(row),
        SemijoinI64Array::I64(array) => array.is_valid(row),
    }
}

pub(super) fn insert_semijoin_i64_utf8_pairs(
    values: &mut SemijoinI64Utf8PairValues,
    numeric_values: &mut SemijoinNumericValues,
    string_ids: &mut FastHashMap<Vec<u8>, u32>,
    left: &ArrayRef,
    right: &ArrayRef,
    numeric_left: bool,
) -> Result<()> {
    let (numbers, string_array, dictionary) = if numeric_left {
        let Some(numbers) = semijoin_i64_array(left) else {
            return Ok(());
        };
        let strings = right.as_any().downcast_ref::<StringArray>();
        let dictionary = semijoin_dictionary_i32_view(right);
        if strings.is_none() && dictionary.is_none() {
            return Ok(());
        }
        (numbers, strings, dictionary)
    } else {
        let Some(numbers) = semijoin_i64_array(right) else {
            return Ok(());
        };
        let strings = left.as_any().downcast_ref::<StringArray>();
        let dictionary = semijoin_dictionary_i32_view(left);
        if strings.is_none() && dictionary.is_none() {
            return Ok(());
        }
        (numbers, strings, dictionary)
    };
    if let Some(strings) = string_array {
        values.reserve(strings.len());
        numeric_values.reserve(strings.len());
        for row in 0..strings.len() {
            if semijoin_i64_array_is_valid(&numbers, row) && strings.is_valid(row) {
                let number = semijoin_i64_array_value(&numbers, row);
                let string_id =
                    semijoin_intern_string_id(string_ids, strings.value(row).as_bytes())?;
                numeric_values.insert_i64(number);
                values.insert(number, string_id);
            }
        }
        return Ok(());
    }
    let Some(dictionary) = dictionary else {
        return Ok(());
    };
    let Some(local_string_ids) = semijoin_dictionary_global_string_ids(dictionary, string_ids)?
    else {
        return Ok(());
    };
    let keys = dictionary.keys();
    values.reserve(keys.len());
    numeric_values.reserve(keys.len());
    for row in 0..keys.len() {
        if semijoin_i64_array_is_valid(&numbers, row) && !dictionary.is_null(row) {
            let Ok(local_id) = usize::try_from(keys[row]) else {
                continue;
            };
            if let Some(Some(string_id)) = local_string_ids.get(local_id) {
                let number = semijoin_i64_array_value(&numbers, row);
                numeric_values.insert_i64(number);
                values.insert(number, *string_id);
            }
        }
    }
    Ok(())
}

pub(super) fn semijoin_dictionary_i32_view(array: &ArrayRef) -> Option<DictionaryI32View<'_>> {
    array
        .as_any()
        .downcast_ref::<DictionaryArray<Int32Type>>()
        .map(DictionaryI32View::Arrow)
}

pub(super) fn semijoin_intern_string_id(
    string_ids: &mut FastHashMap<Vec<u8>, u32>,
    value: &[u8],
) -> Result<u32> {
    if let Some(id) = string_ids.get(value) {
        return Ok(*id);
    }
    let id = u32::try_from(string_ids.len()).map_err(|_| {
        DodamError::UnsupportedSql("too many distinct string semijoin keys".to_string())
    })?;
    string_ids.insert(value.to_vec(), id);
    Ok(id)
}

pub(super) fn semijoin_dictionary_global_string_ids(
    dictionary: DictionaryI32View<'_>,
    string_ids: &mut FastHashMap<Vec<u8>, u32>,
) -> Result<Option<Vec<Option<u32>>>> {
    let Some(values) = dictionary.string_values() else {
        return Ok(None);
    };
    let mut ids = Vec::with_capacity(values.len());
    for index in 0..values.len() {
        ids.push(Some(semijoin_intern_string_id(
            string_ids,
            values.value_bytes(index),
        )?));
    }
    Ok(Some(ids))
}

pub(super) fn semijoin_dictionary_existing_string_ids(
    dictionary: DictionaryI32View<'_>,
    string_ids: &FastHashMap<Vec<u8>, u32>,
) -> Option<Vec<Option<u32>>> {
    let values = dictionary.string_values()?;
    let mut ids = Vec::with_capacity(values.len());
    for index in 0..values.len() {
        ids.push(string_ids.get(values.value_bytes(index)).copied());
    }
    Some(ids)
}

pub(super) fn semijoin_i64_utf8_pair_mask(
    batch: &RecordBatch,
    left_column: &str,
    right_column: &str,
    keys: &SemijoinI64Utf8PairKeys,
    negated: bool,
) -> Result<BooleanArray> {
    let left_index = batch_column_index(batch, left_column)?;
    let right_index = batch_column_index(batch, right_column)?;
    let left = batch.column(left_index);
    let right = batch.column(right_index);
    let (numbers, string_array, dictionary) = if keys.numeric_left {
        let Some(numbers) = semijoin_i64_array(left) else {
            return Err(DodamError::UnsupportedSql(
                "integer/string semijoin key type changed across batches".to_string(),
            ));
        };
        let strings = right.as_any().downcast_ref::<StringArray>();
        let dictionary = semijoin_dictionary_i32_view(right);
        if strings.is_none() && dictionary.is_none() {
            return Err(DodamError::UnsupportedSql(
                "integer/string semijoin key type changed across batches".to_string(),
            ));
        }
        (numbers, strings, dictionary)
    } else {
        let Some(numbers) = semijoin_i64_array(right) else {
            return Err(DodamError::UnsupportedSql(
                "integer/string semijoin key type changed across batches".to_string(),
            ));
        };
        let strings = left.as_any().downcast_ref::<StringArray>();
        let dictionary = semijoin_dictionary_i32_view(left);
        if strings.is_none() && dictionary.is_none() {
            return Err(DodamError::UnsupportedSql(
                "integer/string semijoin key type changed across batches".to_string(),
            ));
        }
        (numbers, strings, dictionary)
    };
    if let Some(dictionary) = dictionary {
        let local_string_ids =
            semijoin_dictionary_existing_string_ids(dictionary, &keys.string_ids);
        if let Some(local_string_ids) = local_string_ids {
            let dictionary_keys = dictionary.keys();
            return Ok(semijoin_i64_utf8_dictionary_pair_mask(
                &numbers,
                dictionary,
                dictionary_keys,
                &local_string_ids,
                keys,
                negated,
                batch.num_rows(),
            ));
        }
    }
    let Some(strings) = string_array else {
        return Err(DodamError::UnsupportedSql(
            "integer/string semijoin dictionary values must be Utf8".to_string(),
        ));
    };
    Ok(semijoin_i64_utf8_plain_pair_mask(
        &numbers,
        strings,
        keys,
        negated,
        batch.num_rows(),
    ))
}

pub(super) fn semijoin_i64_utf8_dictionary_pair_mask(
    numbers: &SemijoinI64Array<'_>,
    dictionary: DictionaryI32View<'_>,
    dictionary_keys: &[i32],
    local_string_ids: &[Option<u32>],
    keys: &SemijoinI64Utf8PairKeys,
    negated: bool,
    rows: usize,
) -> BooleanArray {
    match numbers {
        SemijoinI64Array::I32(values) => semijoin_i32_utf8_dictionary_pair_mask(
            values,
            dictionary,
            dictionary_keys,
            local_string_ids,
            keys,
            negated,
            rows,
        ),
        SemijoinI64Array::I64(values) => semijoin_i64_utf8_dictionary_pair_mask_typed(
            values,
            dictionary,
            dictionary_keys,
            local_string_ids,
            keys,
            negated,
            rows,
        ),
    }
}

pub(super) fn semijoin_i32_utf8_dictionary_pair_mask(
    numbers: &Int32Array,
    dictionary: DictionaryI32View<'_>,
    dictionary_keys: &[i32],
    local_string_ids: &[Option<u32>],
    keys: &SemijoinI64Utf8PairKeys,
    negated: bool,
    rows: usize,
) -> BooleanArray {
    if numbers.null_count() == 0 && dictionary.null_count() == 0 {
        return semijoin_i32_utf8_dictionary_pair_mask_null_free(
            numbers.values().as_ref(),
            dictionary_keys,
            local_string_ids,
            keys,
            negated,
            rows,
        );
    }
    let mut selected = BooleanBufferBuilder::new(rows);
    for row in 0..rows {
        if numbers.is_null(row) || dictionary.is_null(row) {
            selected.append(false);
            continue;
        }
        let number = numbers.value(row);
        if !keys.numeric_values.contains_i32(number) {
            selected.append(negated);
            continue;
        }
        let number = i64::from(number);
        let matched = usize::try_from(dictionary_keys[row])
            .ok()
            .and_then(|local_id| local_string_ids.get(local_id).copied().flatten())
            .is_some_and(|string_id| keys.values.contains(number, string_id));
        selected.append(if negated { !matched } else { matched });
    }
    BooleanArray::new(selected.finish(), None)
}

pub(super) fn semijoin_i32_utf8_dictionary_pair_mask_null_free(
    numbers: &[i32],
    dictionary_keys: &[i32],
    local_string_ids: &[Option<u32>],
    keys: &SemijoinI64Utf8PairKeys,
    negated: bool,
    rows: usize,
) -> BooleanArray {
    let mut selected = BooleanBufferBuilder::new(rows);
    for row in 0..rows {
        let number = numbers[row];
        if !keys.numeric_values.contains_i32(number) {
            selected.append(negated);
            continue;
        }
        let number = i64::from(number);
        let matched = usize::try_from(dictionary_keys[row])
            .ok()
            .and_then(|local_id| local_string_ids.get(local_id).copied().flatten())
            .is_some_and(|string_id| keys.values.contains(number, string_id));
        selected.append(if negated { !matched } else { matched });
    }
    BooleanArray::new(selected.finish(), None)
}

pub(super) fn semijoin_i64_utf8_dictionary_pair_mask_typed(
    numbers: &Int64Array,
    dictionary: DictionaryI32View<'_>,
    dictionary_keys: &[i32],
    local_string_ids: &[Option<u32>],
    keys: &SemijoinI64Utf8PairKeys,
    negated: bool,
    rows: usize,
) -> BooleanArray {
    if numbers.null_count() == 0 && dictionary.null_count() == 0 {
        return semijoin_i64_utf8_dictionary_pair_mask_null_free(
            numbers.values().as_ref(),
            dictionary_keys,
            local_string_ids,
            keys,
            negated,
            rows,
        );
    }
    let mut selected = BooleanBufferBuilder::new(rows);
    for row in 0..rows {
        if numbers.is_null(row) || dictionary.is_null(row) {
            selected.append(false);
            continue;
        }
        let number = numbers.value(row);
        if !keys.numeric_values.contains_i64(number) {
            selected.append(negated);
            continue;
        }
        let matched = usize::try_from(dictionary_keys[row])
            .ok()
            .and_then(|local_id| local_string_ids.get(local_id).copied().flatten())
            .is_some_and(|string_id| keys.values.contains(number, string_id));
        selected.append(if negated { !matched } else { matched });
    }
    BooleanArray::new(selected.finish(), None)
}

pub(super) fn semijoin_i64_utf8_dictionary_pair_mask_null_free(
    numbers: &[i64],
    dictionary_keys: &[i32],
    local_string_ids: &[Option<u32>],
    keys: &SemijoinI64Utf8PairKeys,
    negated: bool,
    rows: usize,
) -> BooleanArray {
    let mut selected = BooleanBufferBuilder::new(rows);
    for row in 0..rows {
        let number = numbers[row];
        if !keys.numeric_values.contains_i64(number) {
            selected.append(negated);
            continue;
        }
        let matched = usize::try_from(dictionary_keys[row])
            .ok()
            .and_then(|local_id| local_string_ids.get(local_id).copied().flatten())
            .is_some_and(|string_id| keys.values.contains(number, string_id));
        selected.append(if negated { !matched } else { matched });
    }
    BooleanArray::new(selected.finish(), None)
}

pub(super) fn semijoin_i64_utf8_plain_pair_mask(
    numbers: &SemijoinI64Array<'_>,
    strings: &StringArray,
    keys: &SemijoinI64Utf8PairKeys,
    negated: bool,
    rows: usize,
) -> BooleanArray {
    match numbers {
        SemijoinI64Array::I32(values) => {
            semijoin_i32_utf8_plain_pair_mask(values, strings, keys, negated, rows)
        }
        SemijoinI64Array::I64(values) => {
            semijoin_i64_utf8_plain_pair_mask_typed(values, strings, keys, negated, rows)
        }
    }
}

pub(super) fn semijoin_i32_utf8_plain_pair_mask(
    numbers: &Int32Array,
    strings: &StringArray,
    keys: &SemijoinI64Utf8PairKeys,
    negated: bool,
    rows: usize,
) -> BooleanArray {
    let mut selected = BooleanBufferBuilder::new(rows);
    for row in 0..rows {
        if numbers.is_null(row) || strings.is_null(row) {
            selected.append(false);
            continue;
        }
        let number = numbers.value(row);
        if !keys.numeric_values.contains_i32(number) {
            selected.append(negated);
            continue;
        }
        let number = i64::from(number);
        let matched = keys
            .string_ids
            .get(strings.value(row).as_bytes())
            .is_some_and(|string_id| keys.values.contains(number, *string_id));
        selected.append(if negated { !matched } else { matched });
    }
    BooleanArray::new(selected.finish(), None)
}

pub(super) fn semijoin_i64_utf8_plain_pair_mask_typed(
    numbers: &Int64Array,
    strings: &StringArray,
    keys: &SemijoinI64Utf8PairKeys,
    negated: bool,
    rows: usize,
) -> BooleanArray {
    let mut selected = BooleanBufferBuilder::new(rows);
    for row in 0..rows {
        if numbers.is_null(row) || strings.is_null(row) {
            selected.append(false);
            continue;
        }
        let number = numbers.value(row);
        if !keys.numeric_values.contains_i64(number) {
            selected.append(negated);
            continue;
        }
        let matched = keys
            .string_ids
            .get(strings.value(row).as_bytes())
            .is_some_and(|string_id| keys.values.contains(number, *string_id));
        selected.append(if negated { !matched } else { matched });
    }
    BooleanArray::new(selected.finish(), None)
}

pub(super) fn semijoin_mixed_early_empty_probe_accepts(
    engine: &DodamEngine,
    path: &Path,
    keys: &SemijoinI64Utf8PairKeys,
) -> Result<bool> {
    if std::env::var_os("DODAM_DISABLE_MIXED_TUPLE_EARLY_EMPTY").is_some() {
        return Ok(false);
    }
    let total_rows = engine.parquet_total_row_count(path)?;
    if total_rows == 0 {
        return Ok(true);
    }
    let ratio = keys.numeric_values.len() as f64 / total_rows as f64;
    let max_pairs = if keys.values.is_dense_i32_to_u32() && keys.string_ids.len() <= 64 {
        mixed_tuple_early_empty_max_dense_pairs()
    } else {
        mixed_tuple_early_empty_max_pairs()
    };
    let accepted =
        ratio <= mixed_tuple_early_empty_max_numeric_ratio() && keys.values.len() <= max_pairs;
    if semijoin_profile_enabled() {
        eprintln!(
            "[dodam:semijoin-profile] mixed-early-empty numeric_values={} pairs={} max_pairs={} total_rows={} ratio={:.6} accepted={}",
            keys.numeric_values.len(),
            keys.values.len(),
            max_pairs,
            total_rows,
            ratio,
            accepted
        );
    }
    Ok(accepted)
}

pub(super) fn semijoin_mixed_prefix_precheck_accepts(
    engine: &DodamEngine,
    path: &Path,
    keys: &SemijoinMixedPrefixCandidateKeys,
) -> Result<bool> {
    let total_rows = engine.parquet_total_row_count(path)?;
    if total_rows == 0 {
        return Ok(true);
    }
    let ratio = keys.numeric_values.len() as f64 / total_rows as f64;
    let accepted = ratio <= mixed_tuple_early_empty_max_numeric_ratio();
    if semijoin_profile_enabled() {
        eprintln!(
            "[dodam:semijoin-profile] mixed-prefix-precheck numeric_values={} total_rows={} ratio={:.6} accepted={}",
            keys.numeric_values.len(),
            total_rows,
            ratio,
            accepted
        );
    }
    Ok(accepted)
}

pub(super) fn mixed_tuple_prefix_precheck_enabled() -> bool {
    std::env::var_os("DODAM_MIXED_TUPLE_PREFIX_PRECHECK").is_some()
        && std::env::var_os("DODAM_DISABLE_MIXED_TUPLE_PREFIX_PRECHECK").is_none()
}

pub(super) fn mixed_tuple_early_empty_max_numeric_ratio() -> f64 {
    std::env::var("DODAM_MIXED_TUPLE_EARLY_EMPTY_MAX_NUMERIC_RATIO")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(0.25)
}

pub(super) fn mixed_tuple_early_empty_max_pairs() -> usize {
    std::env::var("DODAM_MIXED_TUPLE_EARLY_EMPTY_MAX_PAIRS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(4096)
}

pub(super) fn mixed_tuple_early_empty_max_dense_pairs() -> usize {
    std::env::var("DODAM_MIXED_TUPLE_EARLY_EMPTY_MAX_DENSE_PAIRS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(200_000)
}

pub(super) fn direct_mixed_tuple_semijoin_outer_has_match(
    engine: &DodamEngine,
    path: &Path,
    batch_size: usize,
    left_column: &str,
    right_column: &str,
    keys: &SemijoinI64Utf8PairKeys,
    outer_filter: Option<&FilterExpr>,
) -> Result<Option<bool>> {
    let (numeric_column, string_column) = if keys.numeric_left {
        (left_column, right_column)
    } else {
        (right_column, left_column)
    };
    if !semijoin_direct_outer_filter_compatible(outer_filter, string_column) {
        if semijoin_profile_enabled() {
            eprintln!(
                "[dodam:semijoin-profile] mixed-early-empty skip=incompatible_outer_filter filter={:?} string_column={}",
                outer_filter.map(FilterExpr::expr),
                string_column
            );
        }
        return Ok(None);
    }
    let row_groups = (0..engine.parquet_row_group_count(path)?).collect::<Vec<_>>();
    if row_groups.is_empty() {
        return Ok(Some(false));
    }
    let mut found = false;
    let mut string_id_cache = SemijoinDictionaryStringIdCache::default();
    if std::env::var_os("DODAM_DISABLE_MIXED_TUPLE_NUMERIC_SELECTED_OUTER").is_none() {
        let metrics = engine.scan_parquet_i32_byte_array_selected_by_i32(
            path,
            &row_groups,
            [numeric_column, string_column],
            |number| keys.numeric_values.contains_i32(number),
            |numbers, dictionary_ids, dictionary| {
                let selected_string_ids =
                    string_id_cache.refresh_existing_dense(dictionary, &keys.string_ids);
                if numbers.len() != dictionary_ids.len() {
                    return Ok(None);
                }
                for (number, dictionary_id) in
                    numbers.iter().copied().zip(dictionary_ids.iter().copied())
                {
                    if direct_mixed_tuple_semijoin_selected_dictionary_row_matches_dense(
                        number,
                        dictionary_id,
                        selected_string_ids,
                        keys,
                    ) {
                        found = true;
                        return Ok(Some(()));
                    }
                }
                Ok(Some(()))
            },
        )?;
        if metrics.is_some() {
            if semijoin_profile_enabled() {
                eprintln!("[dodam:semijoin-profile] mixed-early-empty found={found}");
            }
            return Ok(Some(found));
        }
    }
    let raw_metrics = engine.scan_parquet_i32_dictionary_id_columns_raw(
        path,
        batch_size,
        &row_groups,
        [numeric_column, string_column],
        |number_bytes, _records, dictionary_def_levels, dictionary_ids, dictionary| {
            let selected_string_ids =
                string_id_cache.refresh_existing(dictionary, &keys.string_ids);
            let mut dictionary_value_offset = 0usize;
            match dictionary_def_levels {
                Some(levels) => {
                    for (row, level) in levels.iter().copied().enumerate() {
                        if level == 0 {
                            continue;
                        }
                        let Some(dictionary_id) = dictionary_ids.get(dictionary_value_offset)
                        else {
                            return Ok(None);
                        };
                        dictionary_value_offset += 1;
                        if direct_mixed_tuple_semijoin_dictionary_row_matches(
                            read_i32_le_unchecked(number_bytes, row),
                            *dictionary_id,
                            &selected_string_ids,
                            keys,
                        ) {
                            found = true;
                            return Ok(Some(()));
                        }
                    }
                }
                None => {
                    for (row, dictionary_id) in dictionary_ids.iter().copied().enumerate() {
                        if direct_mixed_tuple_semijoin_dictionary_row_matches(
                            read_i32_le_unchecked(number_bytes, row),
                            dictionary_id,
                            &selected_string_ids,
                            keys,
                        ) {
                            found = true;
                            return Ok(Some(()));
                        }
                    }
                    dictionary_value_offset = dictionary_ids.len();
                }
            }
            if dictionary_value_offset != dictionary_ids.len() {
                return Ok(None);
            }
            Ok(Some(()))
        },
    )?;
    let metrics = if raw_metrics.is_some() {
        raw_metrics
    } else {
        engine.scan_parquet_i32_dictionary_id_columns(
            path,
            batch_size,
            &row_groups,
            [numeric_column, string_column],
            |numbers, dictionary_def_levels, dictionary_ids, dictionary| {
                let selected_string_ids =
                    string_id_cache.refresh_existing(dictionary, &keys.string_ids);
                let mut dictionary_value_offset = 0usize;
                match dictionary_def_levels {
                    Some(levels) => {
                        for (row, level) in levels.iter().copied().enumerate() {
                            if level == 0 {
                                continue;
                            }
                            let Some(dictionary_id) = dictionary_ids.get(dictionary_value_offset)
                            else {
                                return Ok(None);
                            };
                            dictionary_value_offset += 1;
                            if direct_mixed_tuple_semijoin_dictionary_row_matches(
                                numbers[row],
                                *dictionary_id,
                                &selected_string_ids,
                                keys,
                            ) {
                                found = true;
                                return Ok(Some(()));
                            }
                        }
                    }
                    None => {
                        for (row, dictionary_id) in dictionary_ids.iter().copied().enumerate() {
                            if direct_mixed_tuple_semijoin_dictionary_row_matches(
                                numbers[row],
                                dictionary_id,
                                &selected_string_ids,
                                keys,
                            ) {
                                found = true;
                                return Ok(Some(()));
                            }
                        }
                        dictionary_value_offset = dictionary_ids.len();
                    }
                }
                if dictionary_value_offset != dictionary_ids.len() {
                    return Ok(None);
                }
                Ok(Some(()))
            },
        )?
    };
    if metrics.is_none() {
        let metrics = engine.scan_parquet_i32_byte_array_columns(
            path,
            batch_size,
            &row_groups,
            [numeric_column, string_column],
            |numbers, string_def_levels, strings| {
                let mut string_value_offset = 0usize;
                for (row, level) in string_def_levels.iter().copied().enumerate() {
                    if level == 0 {
                        continue;
                    }
                    let Some(value) = strings.get(string_value_offset) else {
                        return Ok(None);
                    };
                    string_value_offset += 1;
                    let number = i64::from(numbers[row]);
                    if !keys.numeric_values.contains_i64(number) {
                        continue;
                    }
                    if keys
                        .string_ids
                        .get(value.as_ref())
                        .is_some_and(|string_id| keys.values.contains(number, *string_id))
                    {
                        found = true;
                        return Ok(Some(()));
                    }
                }
                if string_value_offset != strings.len() {
                    return Ok(None);
                }
                Ok(Some(()))
            },
        )?;
        if metrics.is_none() {
            if semijoin_profile_enabled() {
                eprintln!(
                    "[dodam:semijoin-profile] mixed-early-empty skip=direct_reader_unsupported"
                );
            }
            return Ok(None);
        }
    }
    if semijoin_profile_enabled() {
        eprintln!("[dodam:semijoin-profile] mixed-early-empty found={found}");
    }
    Ok(Some(found))
}

pub(super) fn direct_mixed_tuple_semijoin_outer_has_prefix_candidate(
    engine: &DodamEngine,
    path: &Path,
    batch_size: usize,
    left_column: &str,
    right_column: &str,
    keys: &SemijoinMixedPrefixCandidateKeys,
    outer_filter: Option<&FilterExpr>,
) -> Result<Option<bool>> {
    let (numeric_column, string_column) = if keys.numeric_left {
        (left_column, right_column)
    } else {
        (right_column, left_column)
    };
    if keys.numeric_values.len() == 0 || keys.string_ids.is_empty() {
        return Ok(Some(false));
    }
    if !semijoin_direct_outer_filter_compatible(outer_filter, string_column) {
        return Ok(None);
    }
    let row_groups = (0..engine.parquet_row_group_count(path)?).collect::<Vec<_>>();
    if row_groups.is_empty() {
        return Ok(Some(false));
    }
    let mut found = false;
    let mut string_id_cache = SemijoinDictionaryStringIdCache::default();
    if std::env::var_os("DODAM_DISABLE_MIXED_TUPLE_NUMERIC_SELECTED_OUTER").is_none() {
        let metrics = engine.scan_parquet_i32_byte_array_selected_by_i32(
            path,
            &row_groups,
            [numeric_column, string_column],
            |number| keys.numeric_values.contains_i32(number),
            |numbers, dictionary_ids, dictionary| {
                let selected_string_ids =
                    string_id_cache.refresh_existing(dictionary, &keys.string_ids);
                if numbers.len() != dictionary_ids.len() {
                    return Ok(None);
                }
                for dictionary_id in dictionary_ids.iter().copied() {
                    if usize::try_from(dictionary_id)
                        .ok()
                        .and_then(|index| selected_string_ids.get(index))
                        .and_then(|id| *id)
                        .is_some()
                    {
                        found = true;
                        return Ok(Some(()));
                    }
                }
                Ok(Some(()))
            },
        )?;
        if metrics.is_some() {
            if semijoin_profile_enabled() {
                eprintln!("[dodam:semijoin-profile] mixed-prefix-precheck outer_found={found}");
            }
            return Ok(Some(found));
        }
    }
    let metrics = engine.scan_parquet_i32_dictionary_id_columns(
        path,
        batch_size,
        &row_groups,
        [numeric_column, string_column],
        |numbers, dictionary_def_levels, dictionary_ids, dictionary| {
            let selected_string_ids =
                string_id_cache.refresh_existing(dictionary, &keys.string_ids);
            let mut dictionary_value_offset = 0usize;
            match dictionary_def_levels {
                Some(levels) => {
                    for (row, level) in levels.iter().copied().enumerate() {
                        if level == 0 {
                            continue;
                        }
                        let Some(dictionary_id) = dictionary_ids.get(dictionary_value_offset)
                        else {
                            return Ok(None);
                        };
                        dictionary_value_offset += 1;
                        if !keys.numeric_values.contains_i32(numbers[row]) {
                            continue;
                        }
                        if usize::try_from(*dictionary_id)
                            .ok()
                            .and_then(|index| selected_string_ids.get(index))
                            .and_then(|id| *id)
                            .is_some()
                        {
                            found = true;
                            return Ok(Some(()));
                        }
                    }
                }
                None => {
                    for (row, dictionary_id) in dictionary_ids.iter().copied().enumerate() {
                        if !keys.numeric_values.contains_i32(numbers[row]) {
                            continue;
                        }
                        if usize::try_from(dictionary_id)
                            .ok()
                            .and_then(|index| selected_string_ids.get(index))
                            .and_then(|id| *id)
                            .is_some()
                        {
                            found = true;
                            return Ok(Some(()));
                        }
                    }
                    dictionary_value_offset = dictionary_ids.len();
                }
            }
            if dictionary_value_offset != dictionary_ids.len() {
                return Ok(None);
            }
            Ok(Some(()))
        },
    )?;
    if metrics.is_none() {
        return Ok(None);
    }
    if semijoin_profile_enabled() {
        eprintln!("[dodam:semijoin-profile] mixed-prefix-precheck outer_found={found}");
    }
    Ok(Some(found))
}

pub(super) fn direct_mixed_tuple_semijoin_dictionary_row_matches(
    number: i32,
    dictionary_id: i32,
    selected_string_ids: &[Option<u32>],
    keys: &SemijoinI64Utf8PairKeys,
) -> bool {
    if !keys.numeric_values.contains_i32(number) {
        return false;
    }
    let number = i64::from(number);
    usize::try_from(dictionary_id)
        .ok()
        .and_then(|dictionary_id| selected_string_ids.get(dictionary_id).copied().flatten())
        .is_some_and(|string_id| keys.values.contains(number, string_id))
}

pub(super) fn direct_mixed_tuple_semijoin_selected_dictionary_row_matches_dense(
    number: i32,
    dictionary_id: i32,
    selected_string_ids: &[u32],
    keys: &SemijoinI64Utf8PairKeys,
) -> bool {
    let Ok(dictionary_id) = usize::try_from(dictionary_id) else {
        return false;
    };
    let Some(&string_id) = selected_string_ids.get(dictionary_id) else {
        return false;
    };
    string_id != u32::MAX && keys.values.contains(i64::from(number), string_id)
}

pub(super) fn semijoin_direct_outer_filter_compatible(
    filter: Option<&FilterExpr>,
    string_column: &str,
) -> bool {
    match filter.map(FilterExpr::expr) {
        None => true,
        Some(Expr::IsNull {
            column,
            negated: true,
        }) => column == string_column,
        _ => false,
    }
}

pub(super) fn semijoin_literal_pair_mask(
    batch: &RecordBatch,
    left_column: &str,
    right_column: &str,
    keys: &SemijoinLiteralPairKeys,
    negated: bool,
) -> Result<BooleanArray> {
    let left_index = batch_column_index(batch, left_column)?;
    let right_index = batch_column_index(batch, right_column)?;
    let left = batch.column(left_index);
    let right = batch.column(right_index);
    let values = (0..batch.num_rows())
        .map(|row| {
            let (Some(left), Some(right)) =
                (semijoin_key_at(left, row)?, semijoin_key_at(right, row)?)
            else {
                return Ok(Some(false));
            };
            let matched = keys.values.contains(&(left, right));
            Ok(Some(if negated { !matched } else { matched }))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(BooleanArray::from(values))
}

pub(super) fn semijoin_i64_pair_mask(
    batch: &RecordBatch,
    left_column: &str,
    right_column: &str,
    keys: &SemijoinI64PairKeys,
    negated: bool,
) -> Result<BooleanArray> {
    let left_index = batch_column_index(batch, left_column)?;
    let right_index = batch_column_index(batch, right_column)?;
    let left = batch.column(left_index);
    let right = batch.column(right_index);
    if negated {
        semijoin_i64_pair_anti_membership_mask_for_arrays(left, right, &keys.values)
    } else {
        semijoin_i64_pair_membership_mask_for_arrays(left, right, &keys.values)
    }
}

pub(super) struct SemijoinI64PairKeys {
    values: SemijoinI64PairSet,
    has_null: bool,
}

pub(super) enum SemijoinI64PairSet {
    Pair(FastHashSet<(i64, i64)>),
    PackedI32(FastHashSet<u64>),
    DenseI32ToI32 {
        min: i32,
        values: Vec<i32>,
        present: Vec<u8>,
    },
}

impl SemijoinI64PairSet {
    fn for_arrays(left: &ArrayRef, right: &ArrayRef) -> Self {
        if left.as_any().is::<Int32Array>() && right.as_any().is::<Int32Array>() {
            Self::PackedI32(FastHashSet::default())
        } else {
            Self::empty_pair()
        }
    }

    fn empty_pair() -> Self {
        Self::Pair(FastHashSet::default())
    }

    fn insert_from_arrays(&mut self, left: &ArrayRef, right: &ArrayRef) -> Result<()> {
        match self {
            Self::DenseI32ToI32 { .. } => Err(DodamError::UnsupportedSql(
                "dense integer semijoin set cannot be extended from Arrow arrays".to_string(),
            )),
            Self::PackedI32(values) => {
                if let (Some(left), Some(right)) = (
                    left.as_any().downcast_ref::<Int32Array>(),
                    right.as_any().downcast_ref::<Int32Array>(),
                ) {
                    insert_semijoin_packed_i32_pairs(values, left, right);
                    return Ok(());
                }
                let mut pairs = FastHashSet::default();
                insert_semijoin_i64_pairs_from_arrays(&mut pairs, left, right)?;
                *self = Self::Pair(pairs);
                Ok(())
            }
            Self::Pair(values) => insert_semijoin_i64_pairs_from_arrays(values, left, right),
        }
    }
}

#[inline]
pub(super) fn pack_i32_pair(left: i32, right: i32) -> u64 {
    (u64::from(left as u32) << 32) | u64::from(right as u32)
}

pub(super) fn insert_semijoin_packed_i32_pairs(
    values: &mut FastHashSet<u64>,
    left: &Int32Array,
    right: &Int32Array,
) {
    if left.null_count() == 0 && right.null_count() == 0 {
        values.reserve(left.len());
        for row in 0..left.len() {
            values.insert(pack_i32_pair(left.value(row), right.value(row)));
        }
        return;
    }
    for row in 0..left.len() {
        if left.is_valid(row) && right.is_valid(row) {
            values.insert(pack_i32_pair(left.value(row), right.value(row)));
        }
    }
}

pub(super) fn insert_semijoin_i64_pairs_from_arrays(
    values: &mut FastHashSet<(i64, i64)>,
    left: &ArrayRef,
    right: &ArrayRef,
) -> Result<()> {
    if let (Some(left), Some(right)) = (
        left.as_any().downcast_ref::<Int32Array>(),
        right.as_any().downcast_ref::<Int32Array>(),
    ) {
        insert_semijoin_i32_i32_pairs(values, left, right);
        return Ok(());
    }
    if let (Some(left), Some(right)) = (
        left.as_any().downcast_ref::<Int32Array>(),
        right.as_any().downcast_ref::<Int64Array>(),
    ) {
        insert_semijoin_i32_i64_pairs(values, left, right);
        return Ok(());
    }
    if let (Some(left), Some(right)) = (
        left.as_any().downcast_ref::<Int64Array>(),
        right.as_any().downcast_ref::<Int32Array>(),
    ) {
        insert_semijoin_i64_i32_pairs(values, left, right);
        return Ok(());
    }
    if let (Some(left), Some(right)) = (
        left.as_any().downcast_ref::<Int64Array>(),
        right.as_any().downcast_ref::<Int64Array>(),
    ) {
        insert_semijoin_i64_i64_pairs(values, left, right);
        return Ok(());
    }
    for row in 0..left.len() {
        let Some(left) = semijoin_i64_key_at(left, row)? else {
            continue;
        };
        let Some(right) = semijoin_i64_key_at(right, row)? else {
            continue;
        };
        values.insert((left, right));
    }
    Ok(())
}

pub(super) fn insert_semijoin_i32_i32_pairs(
    values: &mut FastHashSet<(i64, i64)>,
    left: &Int32Array,
    right: &Int32Array,
) {
    if left.null_count() == 0 && right.null_count() == 0 {
        values.reserve(left.len());
        for row in 0..left.len() {
            values.insert((i64::from(left.value(row)), i64::from(right.value(row))));
        }
        return;
    }
    for row in 0..left.len() {
        if left.is_valid(row) && right.is_valid(row) {
            values.insert((i64::from(left.value(row)), i64::from(right.value(row))));
        }
    }
}

pub(super) fn insert_semijoin_i32_i64_pairs(
    values: &mut FastHashSet<(i64, i64)>,
    left: &Int32Array,
    right: &Int64Array,
) {
    if left.null_count() == 0 && right.null_count() == 0 {
        values.reserve(left.len());
        for row in 0..left.len() {
            values.insert((i64::from(left.value(row)), right.value(row)));
        }
        return;
    }
    for row in 0..left.len() {
        if left.is_valid(row) && right.is_valid(row) {
            values.insert((i64::from(left.value(row)), right.value(row)));
        }
    }
}

pub(super) fn insert_semijoin_i64_i32_pairs(
    values: &mut FastHashSet<(i64, i64)>,
    left: &Int64Array,
    right: &Int32Array,
) {
    if left.null_count() == 0 && right.null_count() == 0 {
        values.reserve(left.len());
        for row in 0..left.len() {
            values.insert((left.value(row), i64::from(right.value(row))));
        }
        return;
    }
    for row in 0..left.len() {
        if left.is_valid(row) && right.is_valid(row) {
            values.insert((left.value(row), i64::from(right.value(row))));
        }
    }
}

pub(super) fn insert_semijoin_i64_i64_pairs(
    values: &mut FastHashSet<(i64, i64)>,
    left: &Int64Array,
    right: &Int64Array,
) {
    if left.null_count() == 0 && right.null_count() == 0 {
        values.reserve(left.len());
        for row in 0..left.len() {
            values.insert((left.value(row), right.value(row)));
        }
        return;
    }
    for row in 0..left.len() {
        if left.is_valid(row) && right.is_valid(row) {
            values.insert((left.value(row), right.value(row)));
        }
    }
}

pub(super) fn semijoin_i64_pair_membership_mask_for_arrays(
    left: &ArrayRef,
    right: &ArrayRef,
    keys: &SemijoinI64PairSet,
) -> Result<BooleanArray> {
    if let SemijoinI64PairSet::DenseI32ToI32 {
        min,
        values,
        present,
    } = keys
        && let (Some(left), Some(right)) = (
            left.as_any().downcast_ref::<Int32Array>(),
            right.as_any().downcast_ref::<Int32Array>(),
        )
    {
        return Ok(semijoin_dense_i32_to_i32_membership_mask(
            left, right, *min, values, present,
        ));
    }
    if let SemijoinI64PairSet::PackedI32(keys) = keys
        && let (Some(left), Some(right)) = (
            left.as_any().downcast_ref::<Int32Array>(),
            right.as_any().downcast_ref::<Int32Array>(),
        )
    {
        return Ok(semijoin_packed_i32_pair_membership_mask(left, right, keys));
    }
    let SemijoinI64PairSet::Pair(keys) = keys else {
        return Err(DodamError::UnsupportedSql(
            "integer semijoin key type changed across batches".to_string(),
        ));
    };
    if let (Some(left), Some(right)) = (
        left.as_any().downcast_ref::<Int32Array>(),
        right.as_any().downcast_ref::<Int32Array>(),
    ) {
        return Ok(semijoin_i32_i32_pair_membership_mask(left, right, keys));
    }
    if let (Some(left), Some(right)) = (
        left.as_any().downcast_ref::<Int32Array>(),
        right.as_any().downcast_ref::<Int64Array>(),
    ) {
        return Ok(semijoin_i32_i64_pair_membership_mask(left, right, keys));
    }
    if let (Some(left), Some(right)) = (
        left.as_any().downcast_ref::<Int64Array>(),
        right.as_any().downcast_ref::<Int32Array>(),
    ) {
        return Ok(semijoin_i64_i32_pair_membership_mask(left, right, keys));
    }
    if let (Some(left), Some(right)) = (
        left.as_any().downcast_ref::<Int64Array>(),
        right.as_any().downcast_ref::<Int64Array>(),
    ) {
        return Ok(semijoin_i64_i64_pair_membership_mask(left, right, keys));
    }
    let values = (0..left.len())
        .map(|row| {
            let Some(left) = semijoin_i64_key_at(left, row)? else {
                return Ok(Some(false));
            };
            let Some(right) = semijoin_i64_key_at(right, row)? else {
                return Ok(Some(false));
            };
            Ok(Some(keys.contains(&(left, right))))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(BooleanArray::from(values))
}

pub(super) fn semijoin_dense_i32_to_i32_membership_mask(
    left: &Int32Array,
    right: &Int32Array,
    min: i32,
    values: &[i32],
    present: &[u8],
) -> BooleanArray {
    let mut output = Vec::with_capacity(left.len());
    for row in 0..left.len() {
        if !left.is_valid(row) || !right.is_valid(row) {
            output.push(Some(false));
            continue;
        }
        let matched = i64::from(left.value(row))
            .checked_sub(i64::from(min))
            .and_then(|offset| usize::try_from(offset).ok())
            .is_some_and(|index| {
                present.get(index).is_some_and(|byte| *byte != 0)
                    && values
                        .get(index)
                        .is_some_and(|value| *value == right.value(row))
            });
        output.push(Some(matched));
    }
    BooleanArray::from(output)
}

pub(super) fn semijoin_i64_pair_anti_membership_mask_for_arrays(
    left: &ArrayRef,
    right: &ArrayRef,
    keys: &SemijoinI64PairSet,
) -> Result<BooleanArray> {
    let membership = semijoin_i64_pair_membership_mask_for_arrays(left, right, keys)?;
    Ok(BooleanArray::from(
        membership
            .iter()
            .enumerate()
            .map(|(row, value)| {
                if left.is_null(row) || right.is_null(row) {
                    Some(false)
                } else {
                    value.map(|matched| !matched)
                }
            })
            .collect::<Vec<_>>(),
    ))
}

pub(super) fn semijoin_packed_i32_pair_membership_mask(
    left: &Int32Array,
    right: &Int32Array,
    keys: &FastHashSet<u64>,
) -> BooleanArray {
    if left.null_count() == 0 && right.null_count() == 0 {
        return boolean_array_no_nulls_from_len(left.len(), |row| {
            keys.contains(&pack_i32_pair(left.value(row), right.value(row)))
        });
    }
    BooleanArray::from(
        (0..left.len())
            .map(|row| {
                (left.is_valid(row) && right.is_valid(row))
                    .then(|| keys.contains(&pack_i32_pair(left.value(row), right.value(row))))
            })
            .collect::<Vec<_>>(),
    )
}

pub(super) fn semijoin_i32_i32_pair_membership_mask(
    left: &Int32Array,
    right: &Int32Array,
    keys: &FastHashSet<(i64, i64)>,
) -> BooleanArray {
    if left.null_count() == 0 && right.null_count() == 0 {
        return boolean_array_no_nulls_from_len(left.len(), |row| {
            keys.contains(&(i64::from(left.value(row)), i64::from(right.value(row))))
        });
    }
    BooleanArray::from(
        (0..left.len())
            .map(|row| {
                (left.is_valid(row) && right.is_valid(row)).then(|| {
                    keys.contains(&(i64::from(left.value(row)), i64::from(right.value(row))))
                })
            })
            .collect::<Vec<_>>(),
    )
}

pub(super) fn semijoin_i32_i64_pair_membership_mask(
    left: &Int32Array,
    right: &Int64Array,
    keys: &FastHashSet<(i64, i64)>,
) -> BooleanArray {
    if left.null_count() == 0 && right.null_count() == 0 {
        return boolean_array_no_nulls_from_len(left.len(), |row| {
            keys.contains(&(i64::from(left.value(row)), right.value(row)))
        });
    }
    BooleanArray::from(
        (0..left.len())
            .map(|row| {
                (left.is_valid(row) && right.is_valid(row))
                    .then(|| keys.contains(&(i64::from(left.value(row)), right.value(row))))
            })
            .collect::<Vec<_>>(),
    )
}

pub(super) fn semijoin_i64_i32_pair_membership_mask(
    left: &Int64Array,
    right: &Int32Array,
    keys: &FastHashSet<(i64, i64)>,
) -> BooleanArray {
    if left.null_count() == 0 && right.null_count() == 0 {
        return boolean_array_no_nulls_from_len(left.len(), |row| {
            keys.contains(&(left.value(row), i64::from(right.value(row))))
        });
    }
    BooleanArray::from(
        (0..left.len())
            .map(|row| {
                (left.is_valid(row) && right.is_valid(row))
                    .then(|| keys.contains(&(left.value(row), i64::from(right.value(row)))))
            })
            .collect::<Vec<_>>(),
    )
}

pub(super) fn semijoin_i64_i64_pair_membership_mask(
    left: &Int64Array,
    right: &Int64Array,
    keys: &FastHashSet<(i64, i64)>,
) -> BooleanArray {
    if left.null_count() == 0 && right.null_count() == 0 {
        return boolean_array_no_nulls_from_len(left.len(), |row| {
            keys.contains(&(left.value(row), right.value(row)))
        });
    }
    BooleanArray::from(
        (0..left.len())
            .map(|row| {
                (left.is_valid(row) && right.is_valid(row))
                    .then(|| keys.contains(&(left.value(row), right.value(row))))
            })
            .collect::<Vec<_>>(),
    )
}
