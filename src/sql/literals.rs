use super::*;

pub(super) fn parse_ymd(value: &str) -> Result<(i32, u32, u32)> {
    let mut parts = value.split('-');
    let year = parts
        .next()
        .and_then(|value| value.parse::<i32>().ok())
        .ok_or_else(|| DodamError::UnsupportedSql(format!("invalid DATE literal: {value}")))?;
    let month = parts
        .next()
        .and_then(|value| value.parse::<u32>().ok())
        .ok_or_else(|| DodamError::UnsupportedSql(format!("invalid DATE literal: {value}")))?;
    let day = parts
        .next()
        .and_then(|value| value.parse::<u32>().ok())
        .ok_or_else(|| DodamError::UnsupportedSql(format!("invalid DATE literal: {value}")))?;
    if parts.next().is_some()
        || !(1..=12).contains(&month)
        || day == 0
        || day > days_in_month(year, month)
    {
        return Err(DodamError::UnsupportedSql(format!(
            "invalid DATE literal: {value}"
        )));
    }
    Ok((year, month, day))
}

pub(super) fn add_months(year: i32, month: u32, day: u32, months: i64) -> Result<(i32, u32, u32)> {
    let month_index = i64::from(year) * 12 + i64::from(month - 1) + months;
    let year = month_index.div_euclid(12);
    let month = month_index.rem_euclid(12) + 1;
    let year = i32::try_from(year)
        .map_err(|_| DodamError::UnsupportedSql("DATE arithmetic overflow".to_string()))?;
    let month = u32::try_from(month)
        .map_err(|_| DodamError::UnsupportedSql("DATE arithmetic overflow".to_string()))?;
    Ok((year, month, day.min(days_in_month(year, month))))
}

fn days_in_month(year: i32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap_year(year) => 29,
        2 => 28,
        _ => 0,
    }
}

fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

pub(super) fn days_from_civil(year: i32, month: u32, day: u32) -> Result<i64> {
    let year = i64::from(year) - i64::from(month <= 2);
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let yoe = year - era * 400;
    let month = i64::from(month);
    let day = i64::from(day);
    let doy = (153 * (month + if month > 2 { -3 } else { 9 }) + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    Ok(era * 146_097 + doe - 719_468)
}

pub(super) fn civil_from_days(days: i64) -> Result<(i32, u32, u32)> {
    let days = days + 719_468;
    let era = if days >= 0 { days } else { days - 146_096 } / 146_097;
    let doe = days - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = mp + if mp < 10 { 3 } else { -9 };
    let year = year + i64::from(month <= 2);
    Ok((
        i32::try_from(year)
            .map_err(|_| DodamError::UnsupportedSql("DATE arithmetic overflow".to_string()))?,
        u32::try_from(month)
            .map_err(|_| DodamError::UnsupportedSql("DATE arithmetic overflow".to_string()))?,
        u32::try_from(day)
            .map_err(|_| DodamError::UnsupportedSql("DATE arithmetic overflow".to_string()))?,
    ))
}

pub(super) fn year_from_days(days: i64) -> Result<i32> {
    let days = days + 719_468;
    let era = if days >= 0 { days } else { days - 146_096 } / 146_097;
    let doe = days - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let month = mp + if mp < 10 { 3 } else { -9 };
    i32::try_from(year + i64::from(month <= 2))
        .map_err(|_| DodamError::UnsupportedSql("DATE arithmetic overflow".to_string()))
}

pub(super) fn parse_usize_literal(expr: &SqlExpr) -> Result<usize> {
    let SqlExpr::Value(value) = expr else {
        return Err(DodamError::UnsupportedSql(format!(
            "expected integer literal, got {expr}"
        )));
    };
    match value.value.clone() {
        Value::Number(value, false) => value.parse::<usize>().map_err(|_| {
            DodamError::UnsupportedSql(format!(
                "expected non-negative integer literal, got {value}"
            ))
        }),
        value => Err(DodamError::UnsupportedSql(format!(
            "expected integer literal, got {value}"
        ))),
    }
}

pub(super) fn parse_decimal_cast_target(target: &str) -> Result<Option<(u8, i8)>> {
    let target = target.trim();
    let Some(args) = target
        .strip_prefix("decimal")
        .or_else(|| target.strip_prefix("numeric"))
    else {
        return Ok(None);
    };
    let args = args.trim();
    if args.is_empty() {
        return Ok(Some((18, 3)));
    }
    let Some(args) = args
        .strip_prefix('(')
        .and_then(|args| args.strip_suffix(')'))
    else {
        return Err(DodamError::InvalidCast(format!(
            "invalid DECIMAL target: {target}"
        )));
    };
    let parts = args.split(',').map(str::trim).collect::<Vec<_>>();
    let [precision, scale] = parts.as_slice() else {
        return Err(DodamError::InvalidCast(format!(
            "invalid DECIMAL target: {target}"
        )));
    };
    let precision = precision
        .parse::<u8>()
        .map_err(|_| DodamError::InvalidCast(format!("invalid DECIMAL precision: {target}")))?;
    let scale = scale
        .parse::<i8>()
        .map_err(|_| DodamError::InvalidCast(format!("invalid DECIMAL scale: {target}")))?;
    if precision == 0 || precision > 38 || scale < 0 || scale > precision as i8 {
        return Err(DodamError::InvalidCast(format!(
            "invalid DECIMAL target: {target}"
        )));
    }
    Ok(Some((precision, scale)))
}

pub(super) fn parse_decimal_literal_to_scaled(
    value: &str,
    scale: i8,
    precision: u8,
) -> Result<i128> {
    let scale_usize = usize::try_from(scale)
        .map_err(|_| DodamError::InvalidCast(format!("invalid decimal scale {scale}")))?;
    let value = value.trim();
    if value.is_empty() {
        return Err(DodamError::InvalidCast(
            "cannot cast empty string to DECIMAL".to_string(),
        ));
    }
    let negative = value.starts_with('-');
    let unsigned = value.strip_prefix(['-', '+']).unwrap_or(value);
    let (whole, fractional) = unsigned.split_once('.').unwrap_or((unsigned, ""));
    if whole.is_empty()
        || !whole.chars().all(|ch| ch.is_ascii_digit())
        || !fractional.chars().all(|ch| ch.is_ascii_digit())
    {
        return Err(DodamError::InvalidCast(format!(
            "cannot cast '{value}' to DECIMAL"
        )));
    }
    let scale_factor = decimal_scale_i128(scale)
        .ok_or_else(|| DodamError::InvalidCast(format!("decimal scale {scale} overflows")))?;
    let mut raw = whole
        .parse::<i128>()
        .map_err(|_| DodamError::InvalidCast(format!("cannot cast '{value}' to DECIMAL")))?
        .checked_mul(scale_factor)
        .ok_or_else(|| DodamError::InvalidCast("decimal cast overflow".to_string()))?;
    let kept = &fractional[..fractional.len().min(scale_usize)];
    let mut kept_value = kept.parse::<i128>().unwrap_or(0);
    for _ in kept.len()..scale_usize {
        kept_value = kept_value
            .checked_mul(10)
            .ok_or_else(|| DodamError::InvalidCast("decimal cast overflow".to_string()))?;
    }
    raw = raw
        .checked_add(kept_value)
        .ok_or_else(|| DodamError::InvalidCast("decimal cast overflow".to_string()))?;
    if fractional.len() > scale_usize {
        let next_digit = fractional.as_bytes()[scale_usize];
        if next_digit >= b'5' {
            raw = raw
                .checked_add(1)
                .ok_or_else(|| DodamError::InvalidCast("decimal cast overflow".to_string()))?;
        }
    }
    if negative {
        raw = -raw;
    }
    validate_decimal_precision(raw, precision)
}

pub(super) fn parse_date32_days(value: &str) -> Result<i32> {
    let (year, month, day) = parse_ymd(value)?;
    i32::try_from(days_from_civil(year, month, day)?)
        .map_err(|_| DodamError::UnsupportedSql("DATE overflow".to_string()))
}

pub(super) fn parse_timestamp_millis_value(value: &str) -> Result<i64> {
    let (date, time) = value
        .split_once(' ')
        .or_else(|| value.split_once('T'))
        .map_or((value, "00:00:00"), |(date, time)| (date, time));
    let (time, timezone_offset_millis) = split_timestamp_timezone(time, value)?;
    let days = i64::from(parse_date32_days(date)?);
    let mut time_parts = time.split(':');
    let hour = parse_timestamp_part(time_parts.next(), value, "hour")?;
    let minute = parse_timestamp_part(time_parts.next(), value, "minute")?;
    let second_raw = time_parts
        .next()
        .ok_or_else(|| DodamError::UnsupportedSql(format!("invalid TIMESTAMP literal: {value}")))?;
    if time_parts.next().is_some() {
        return Err(DodamError::UnsupportedSql(format!(
            "invalid TIMESTAMP literal: {value}"
        )));
    }
    let (second_text, millis_text) = second_raw.split_once('.').unwrap_or((second_raw, ""));
    let second = second_text
        .parse::<i64>()
        .map_err(|_| DodamError::UnsupportedSql(format!("invalid TIMESTAMP literal: {value}")))?;
    let millis = if millis_text.is_empty() {
        0
    } else {
        let millis_text = &millis_text[..millis_text.len().min(3)];
        millis_text.parse::<i64>().map_err(|_| {
            DodamError::UnsupportedSql(format!("invalid TIMESTAMP literal: {value}"))
        })? * 10_i64.pow(u32::try_from(3_usize.saturating_sub(millis_text.len())).unwrap_or(0))
    };
    if !(0..24).contains(&hour) || !(0..60).contains(&minute) || !(0..60).contains(&second) {
        return Err(DodamError::UnsupportedSql(format!(
            "invalid TIMESTAMP literal: {value}"
        )));
    }
    days.checked_mul(86_400_000)
        .and_then(|base| {
            base.checked_add(hour * 3_600_000 + minute * 60_000 + second * 1_000 + millis)
        })
        .and_then(|millis| millis.checked_sub(timezone_offset_millis))
        .ok_or_else(|| DodamError::UnsupportedSql("TIMESTAMP overflow".to_string()))
}

fn parse_timestamp_part(part: Option<&str>, value: &str, label: &str) -> Result<i64> {
    part.ok_or_else(|| DodamError::UnsupportedSql(format!("invalid TIMESTAMP literal: {value}")))?
        .parse::<i64>()
        .map_err(|_| DodamError::UnsupportedSql(format!("invalid TIMESTAMP {label}: {value}")))
}

fn split_timestamp_timezone<'a>(time: &'a str, value: &str) -> Result<(&'a str, i64)> {
    if let Some(stripped) = time.strip_suffix('Z') {
        return Ok((stripped, 0));
    }
    let Some((index, _)) = time
        .char_indices()
        .rev()
        .find(|(index, ch)| *index > 0 && matches!(ch, '+' | '-'))
    else {
        return Ok((time, 0));
    };
    let (time, offset) = time.split_at(index);
    Ok((time, parse_timestamp_timezone_offset(offset, value)?))
}

fn parse_timestamp_timezone_offset(offset: &str, value: &str) -> Result<i64> {
    let Some(sign) = offset.chars().next() else {
        return Err(DodamError::UnsupportedSql(format!(
            "invalid TIMESTAMP literal: {value}"
        )));
    };
    let offset = &offset[sign.len_utf8()..];
    let (hours, minutes) = offset.split_once(':').unwrap_or((offset, "0"));
    let hours = hours
        .parse::<i64>()
        .map_err(|_| DodamError::UnsupportedSql(format!("invalid TIMESTAMP literal: {value}")))?;
    let minutes = minutes
        .parse::<i64>()
        .map_err(|_| DodamError::UnsupportedSql(format!("invalid TIMESTAMP literal: {value}")))?;
    if !(0..=23).contains(&hours) || !(0..=59).contains(&minutes) {
        return Err(DodamError::UnsupportedSql(format!(
            "invalid TIMESTAMP literal: {value}"
        )));
    }
    let millis = hours
        .checked_mul(3_600_000)
        .and_then(|value| value.checked_add(minutes * 60_000))
        .ok_or_else(|| DodamError::UnsupportedSql("TIMESTAMP overflow".to_string()))?;
    Ok(if sign == '-' { -millis } else { millis })
}

pub(super) fn sql_like_pattern(expr: &SqlExpr) -> Result<String> {
    match sql_literal_value(expr)? {
        LiteralValue::Utf8(pattern) => Ok(pattern),
        LiteralValue::Null => Err(DodamError::UnsupportedSql(
            "LIKE NULL patterns are not supported yet".to_string(),
        )),
        value => Err(DodamError::UnsupportedSql(format!(
            "LIKE pattern must be a string literal, got {value}"
        ))),
    }
}

pub(super) fn sql_like_escape(
    escape_char: &Option<sqlparser::ast::ValueWithSpan>,
) -> Result<Option<char>> {
    let Some(escape_char) = escape_char else {
        return Ok(None);
    };
    let Value::SingleQuotedString(value) = &escape_char.value else {
        return Err(DodamError::UnsupportedSql(
            "LIKE ESCAPE must be a string literal".to_string(),
        ));
    };
    let mut chars = value.chars();
    let Some(ch) = chars.next() else {
        return Err(DodamError::UnsupportedSql(
            "LIKE ESCAPE must contain exactly one character".to_string(),
        ));
    };
    if chars.next().is_some() {
        return Err(DodamError::UnsupportedSql(
            "LIKE ESCAPE must contain exactly one character".to_string(),
        ));
    }
    Ok(Some(ch))
}

pub(super) fn sql_literal_value(expr: &SqlExpr) -> Result<LiteralValue> {
    match expr {
        SqlExpr::Value(value) => match &value.value {
            Value::Number(value, false) => value
                .parse::<i64>()
                .map(LiteralValue::Int64)
                .or_else(|_| value.parse::<f64>().map(LiteralValue::Float64))
                .map_err(|_| {
                    DodamError::UnsupportedSql(format!("unsupported numeric literal: {value}"))
                }),
            Value::SingleQuotedString(value) | Value::DoubleQuotedString(value) => {
                Ok(LiteralValue::Utf8(value.clone()))
            }
            Value::Boolean(value) => Ok(LiteralValue::Boolean(*value)),
            Value::Null => Ok(LiteralValue::Null),
            value => Err(DodamError::UnsupportedSql(format!(
                "unsupported literal: {value}"
            ))),
        },
        SqlExpr::TypedString(typed) if typed.data_type.to_string().eq_ignore_ascii_case("date") => {
            match &typed.value.value {
                Value::SingleQuotedString(value) | Value::DoubleQuotedString(value) => {
                    Ok(LiteralValue::Utf8(value.clone()))
                }
                value => Err(DodamError::UnsupportedSql(format!(
                    "unsupported DATE literal: {value}"
                ))),
            }
        }
        SqlExpr::BinaryOp { left, op, right }
            if matches!(op, BinaryOperator::Plus | BinaryOperator::Minus) =>
        {
            if let Some(value) = decimal_number_literal_arithmetic(left, op, right)? {
                return Ok(value);
            }
            let left_value = sql_literal_value(left)?;
            if let Some((amount, field)) = interval_literal(right)? {
                return apply_date_interval(left_value, op.clone(), amount, field);
            }
            let right_value = sql_literal_value(right)?;
            apply_literal_arithmetic(left_value, op.clone(), right_value)
        }
        SqlExpr::UnaryOp { op, expr }
            if matches!(op, UnaryOperator::Minus | UnaryOperator::Plus) =>
        {
            let value = sql_literal_value(expr)?;
            match (op, value) {
                (UnaryOperator::Plus, value) => Ok(value),
                (UnaryOperator::Minus, LiteralValue::Int64(value)) => {
                    Ok(LiteralValue::Int64(value.checked_neg().ok_or_else(
                        || DodamError::UnsupportedSql("integer literal overflow".to_string()),
                    )?))
                }
                (UnaryOperator::Minus, LiteralValue::Float64(value)) => {
                    Ok(LiteralValue::Float64(-value))
                }
                (UnaryOperator::Minus, value) => Err(DodamError::UnsupportedSql(format!(
                    "unary minus requires a numeric literal, got {value}"
                ))),
                _ => unreachable!("validated unary operator"),
            }
        }
        SqlExpr::Substring {
            expr,
            substring_from,
            substring_for,
            ..
        } => {
            let LiteralValue::Utf8(value) = sql_literal_value(expr)? else {
                return Err(DodamError::UnsupportedSql(format!(
                    "SUBSTRING literal input must be a string, got {expr}"
                )));
            };
            let Some(start) = substring_from else {
                return Err(DodamError::UnsupportedSql(
                    "SUBSTRING requires a FROM/start expression".to_string(),
                ));
            };
            let start = literal_usize(start, "SUBSTRING start")?;
            let length = substring_for
                .as_ref()
                .map(|expr| literal_usize(expr, "SUBSTRING length"))
                .transpose()?;
            Ok(LiteralValue::Utf8(substring_literal(&value, start, length)))
        }
        _ => Err(DodamError::UnsupportedSql(format!(
            "expected literal, got {expr}"
        ))),
    }
}

pub(super) fn evaluate_literal_in_list(
    value: &LiteralValue,
    list: &[SqlExpr],
    negated: bool,
) -> Result<Option<bool>> {
    let mut has_null = false;
    for candidate in list {
        let candidate = sql_literal_value(candidate)?;
        if matches!(candidate, LiteralValue::Null) {
            has_null = true;
            continue;
        }
        if compare_literal_values(value, &BinaryOperator::Eq, &candidate)? == Some(true) {
            return Ok(Some(!negated));
        }
    }
    if has_null {
        Ok(None)
    } else {
        Ok(Some(negated))
    }
}

pub(super) fn literal_usize(expr: &SqlExpr, context: &str) -> Result<usize> {
    let LiteralValue::Int64(value) = sql_literal_value(expr)? else {
        return Err(DodamError::UnsupportedSql(format!(
            "{context} must be an integer literal"
        )));
    };
    usize::try_from(value)
        .map_err(|_| DodamError::UnsupportedSql(format!("{context} must be non-negative")))
}

pub(super) fn substring_literal(value: &str, start: usize, length: Option<usize>) -> String {
    let start = start.saturating_sub(1);
    value
        .chars()
        .skip(start)
        .take(length.unwrap_or(usize::MAX))
        .collect()
}

fn decimal_number_literal_arithmetic(
    left: &SqlExpr,
    op: &BinaryOperator,
    right: &SqlExpr,
) -> Result<Option<LiteralValue>> {
    let Some(left) = numeric_literal_text(left) else {
        return Ok(None);
    };
    let Some(right) = numeric_literal_text(right) else {
        return Ok(None);
    };
    let scale = decimal_scale(left).max(decimal_scale(right));
    let left = decimal_literal_to_scaled(left, scale)?;
    let right = decimal_literal_to_scaled(right, scale)?;
    let value = match op {
        BinaryOperator::Plus => left + right,
        BinaryOperator::Minus => left - right,
        _ => unreachable!("validated arithmetic operator"),
    };
    if scale == 0 {
        return Ok(Some(LiteralValue::Int64(i64::try_from(value).map_err(
            |_| DodamError::UnsupportedSql("numeric literal overflow".to_string()),
        )?)));
    }
    let negative = value < 0;
    let value = value.abs();
    let factor =
        10_i128.pow(u32::try_from(scale).map_err(|_| {
            DodamError::UnsupportedSql("numeric literal scale overflow".to_string())
        })?);
    let whole = value / factor;
    let fractional = value % factor;
    let literal = format!(
        "{}{}.{:0width$}",
        if negative { "-" } else { "" },
        whole,
        fractional,
        width = scale
    );
    Ok(Some(LiteralValue::Float64(
        literal.parse::<f64>().map_err(|_| {
            DodamError::UnsupportedSql(format!("unsupported numeric literal: {literal}"))
        })?,
    )))
}

fn numeric_literal_text(expr: &SqlExpr) -> Option<&str> {
    let SqlExpr::Value(value) = expr else {
        return None;
    };
    let Value::Number(value, false) = &value.value else {
        return None;
    };
    Some(value)
}

fn decimal_scale(value: &str) -> usize {
    value
        .split_once('.')
        .map(|(_, fractional)| fractional.len())
        .unwrap_or(0)
}

fn decimal_literal_to_scaled(value: &str, scale: usize) -> Result<i128> {
    let negative = value.starts_with('-');
    let unsigned = value.strip_prefix(['-', '+']).unwrap_or(value);
    let (whole, fractional) = unsigned.split_once('.').unwrap_or((unsigned, ""));
    if whole.is_empty()
        || !whole.chars().all(|ch| ch.is_ascii_digit())
        || !fractional.chars().all(|ch| ch.is_ascii_digit())
        || fractional.len() > scale
    {
        return Err(DodamError::UnsupportedSql(format!(
            "unsupported numeric literal: {value}"
        )));
    }
    let mut result = whole
        .parse::<i128>()
        .map_err(|_| DodamError::UnsupportedSql(format!("unsupported numeric literal: {value}")))?
        .checked_mul(10_i128.pow(u32::try_from(scale).map_err(|_| {
            DodamError::UnsupportedSql("numeric literal scale overflow".to_string())
        })?))
        .ok_or_else(|| DodamError::UnsupportedSql("numeric literal overflow".to_string()))?;
    let mut fractional_value = fractional.parse::<i128>().unwrap_or(0);
    for _ in fractional.len()..scale {
        fractional_value = fractional_value
            .checked_mul(10)
            .ok_or_else(|| DodamError::UnsupportedSql("numeric literal overflow".to_string()))?;
    }
    result = result
        .checked_add(fractional_value)
        .ok_or_else(|| DodamError::UnsupportedSql("numeric literal overflow".to_string()))?;
    if negative {
        result = -result;
    }
    Ok(result)
}

fn apply_literal_arithmetic(
    left: LiteralValue,
    op: BinaryOperator,
    right: LiteralValue,
) -> Result<LiteralValue> {
    match (left, right) {
        (LiteralValue::Int64(left), LiteralValue::Int64(right)) => {
            Ok(LiteralValue::Int64(match op {
                BinaryOperator::Plus => left + right,
                BinaryOperator::Minus => left - right,
                _ => unreachable!("validated arithmetic operator"),
            }))
        }
        (left, right) => {
            let left = literal_as_f64(&left)?;
            let right = literal_as_f64(&right)?;
            Ok(LiteralValue::Float64(match op {
                BinaryOperator::Plus => left + right,
                BinaryOperator::Minus => left - right,
                _ => unreachable!("validated arithmetic operator"),
            }))
        }
    }
}

pub(super) fn literal_as_f64(value: &LiteralValue) -> Result<f64> {
    match value {
        LiteralValue::Int64(value) => Ok(*value as f64),
        LiteralValue::Float64(value) => Ok(*value),
        _ => Err(DodamError::UnsupportedSql(format!(
            "expected numeric literal, got {value}"
        ))),
    }
}

fn interval_literal(expr: &SqlExpr) -> Result<Option<(i64, DateTimeField)>> {
    let SqlExpr::Interval(interval) = expr else {
        return Ok(None);
    };
    let SqlExpr::Value(value) = interval.value.as_ref() else {
        return Err(DodamError::UnsupportedSql(format!(
            "unsupported INTERVAL value: {}",
            interval.value
        )));
    };
    let amount = match &value.value {
        Value::SingleQuotedString(value) | Value::DoubleQuotedString(value) => value,
        value => {
            return Err(DodamError::UnsupportedSql(format!(
                "unsupported INTERVAL literal: {value}"
            )));
        }
    }
    .parse::<i64>()
    .map_err(|_| DodamError::UnsupportedSql(format!("unsupported INTERVAL: {expr}")))?;
    let field = interval.leading_field.clone().ok_or_else(|| {
        DodamError::UnsupportedSql(format!("INTERVAL requires a leading field: {expr}"))
    })?;
    Ok(Some((amount, field)))
}

fn apply_date_interval(
    value: LiteralValue,
    op: BinaryOperator,
    amount: i64,
    field: DateTimeField,
) -> Result<LiteralValue> {
    let LiteralValue::Utf8(date) = value else {
        return Err(DodamError::UnsupportedSql(
            "INTERVAL arithmetic currently requires a DATE literal".to_string(),
        ));
    };
    let amount = match op {
        BinaryOperator::Plus => amount,
        BinaryOperator::Minus => -amount,
        _ => unreachable!("validated arithmetic operator"),
    };
    let (year, month, day) = parse_ymd(&date)?;
    let (year, month, day) = match field {
        DateTimeField::Day => {
            let days = days_from_civil(year, month, day)? + amount;
            civil_from_days(days)?
        }
        DateTimeField::Month => add_months(year, month, day, amount)?,
        DateTimeField::Year => add_months(year, month, day, amount * 12)?,
        field => {
            return Err(DodamError::UnsupportedSql(format!(
                "unsupported INTERVAL field for DATE arithmetic: {field}"
            )));
        }
    };
    Ok(LiteralValue::Utf8(format!("{year:04}-{month:02}-{day:02}")))
}

#[derive(Default)]
pub(super) struct Date32YearCache {
    base_day: i32,
    years: Vec<i32>,
    disabled: bool,
}

impl Date32YearCache {
    const MAX_SPAN_DAYS: usize = 20_000;

    pub(super) fn year(&mut self, day: i32) -> Result<i32> {
        if self.disabled {
            return year_from_days(i64::from(day));
        }
        if self.years.is_empty() {
            self.base_day = day;
            self.years.push(year_from_days(i64::from(day))?);
            return Ok(self.years[0]);
        }
        if day < self.base_day {
            let prepend = usize::try_from(self.base_day - day)
                .map_err(|_| DodamError::UnsupportedSql("DATE arithmetic overflow".to_string()))?;
            if prepend + self.years.len() > Self::MAX_SPAN_DAYS {
                self.disabled = true;
                self.years.clear();
                return year_from_days(i64::from(day));
            }
            let mut years = Vec::with_capacity(prepend + self.years.len());
            for offset in 0..prepend {
                years.push(year_from_days(i64::from(day) + offset as i64)?);
            }
            years.extend_from_slice(&self.years);
            self.base_day = day;
            self.years = years;
            return Ok(self.years[0]);
        }
        let index = usize::try_from(day - self.base_day)
            .map_err(|_| DodamError::UnsupportedSql("DATE arithmetic overflow".to_string()))?;
        if index >= self.years.len() {
            if index + 1 > Self::MAX_SPAN_DAYS {
                self.disabled = true;
                self.years.clear();
                return year_from_days(i64::from(day));
            }
            let start = self.years.len();
            self.years.reserve(index + 1 - start);
            for offset in start..=index {
                self.years
                    .push(year_from_days(i64::from(self.base_day) + offset as i64)?);
            }
        }
        Ok(self.years[index])
    }
}

pub(super) fn sql_comparison_op(op: &BinaryOperator) -> ComparisonOp {
    match op {
        BinaryOperator::Eq => ComparisonOp::Eq,
        BinaryOperator::NotEq => ComparisonOp::NotEq,
        BinaryOperator::Gt => ComparisonOp::Gt,
        BinaryOperator::GtEq => ComparisonOp::GtEq,
        BinaryOperator::Lt => ComparisonOp::Lt,
        BinaryOperator::LtEq => ComparisonOp::LtEq,
        _ => unreachable!("validated comparison operator"),
    }
}
