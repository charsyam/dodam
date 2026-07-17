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
