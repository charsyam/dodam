use super::*;

pub(super) fn format_decimal128_value(value: i128, scale: i8) -> String {
    if scale <= 0 {
        return (value * 10_i128.pow(u32::from(scale.unsigned_abs()))).to_string();
    }
    let scale_u32 = u32::try_from(scale).unwrap_or(0);
    let divisor = 10_i128.pow(scale_u32);
    let negative = value < 0;
    let absolute = value.abs();
    let whole = absolute / divisor;
    let fraction = absolute % divisor;
    let sign = if negative { "-" } else { "" };
    format!(
        "{sign}{whole}.{fraction:0width$}",
        width = usize::try_from(scale_u32).unwrap_or(0)
    )
}

pub(super) fn format_f64_for_sql_varchar(value: f64) -> String {
    if value.is_finite() && value.fract() == 0.0 {
        format!("{value:.1}")
    } else {
        value.to_string()
    }
}

pub(super) fn format_date32_days(days: i32) -> String {
    match civil_from_days(i64::from(days)) {
        Ok((year, month, day)) => format!("{year:04}-{month:02}-{day:02}"),
        Err(_) => days.to_string(),
    }
}

pub(super) fn format_timestamp_millis(millis: i64) -> String {
    let days = millis.div_euclid(86_400_000);
    let millis_of_day = millis.rem_euclid(86_400_000);
    let Ok((year, month, day)) = civil_from_days(days) else {
        return millis.to_string();
    };
    let seconds_of_day = millis_of_day / 1_000;
    let hour = seconds_of_day / 3_600;
    let minute = (seconds_of_day % 3_600) / 60;
    let second = seconds_of_day % 60;
    let millis_remainder = millis_of_day % 1_000;
    if millis_remainder == 0 {
        format!("{year:04}-{month:02}-{day:02} {hour:02}:{minute:02}:{second:02}")
    } else {
        format!(
            "{year:04}-{month:02}-{day:02} {hour:02}:{minute:02}:{second:02}.{millis_remainder:03}"
        )
    }
}

pub(super) fn coalesce_options<T>(left: Vec<Option<T>>, right: Vec<Option<T>>) -> Vec<Option<T>> {
    left.into_iter()
        .zip(right)
        .map(|(left, right)| left.or(right))
        .collect()
}

pub(super) fn rename_output_batches(
    batches: Vec<RecordBatch>,
    aliases: &[(String, String)],
) -> Result<Vec<RecordBatch>> {
    if aliases.is_empty() {
        return Ok(batches);
    }

    batches
        .into_iter()
        .map(|batch| {
            let fields = batch
                .schema()
                .fields()
                .iter()
                .map(|field| {
                    let name = aliases
                        .iter()
                        .find(|(alias, target)| !alias.contains('(') && target == field.name())
                        .map(|(alias, _)| alias.as_str())
                        .unwrap_or_else(|| field.name().as_str());
                    Field::new(name, field.data_type().clone(), field.is_nullable())
                })
                .collect::<Vec<_>>();
            RecordBatch::try_new(Arc::new(Schema::new(fields)), batch.columns().to_vec())
                .map_err(DodamError::from)
        })
        .collect()
}
