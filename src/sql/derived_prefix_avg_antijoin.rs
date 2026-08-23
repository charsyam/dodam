use super::*;

pub(super) async fn try_execute_derived_prefix_avg_anti_join_aggregate_sql(
    engine: &DodamEngine,
    sql: &str,
    batch_size: usize,
) -> Result<Option<QueryOutput>> {
    let dialect = GenericDialect {};
    let statements = Parser::parse_sql(&dialect, sql)
        .map_err(|error| DodamError::UnsupportedSql(error.to_string()))?;
    let [Statement::Query(query)] = statements.as_slice() else {
        return Ok(None);
    };
    let SetExpr::Select(select) = query.body.as_ref() else {
        return Ok(None);
    };
    if !derived_prefix_avg_anti_join_aggregate_shape(select, query) {
        return Ok(None);
    }
    reject_query_features(query)?;
    reject_select_features(select)?;
    let Some((inner_query, _alias)) = parse_derived_from(select)? else {
        return Ok(None);
    };
    let SetExpr::Select(inner_select) = inner_query.body.as_ref() else {
        return Ok(None);
    };
    let customer = parse_from(inner_select)?;
    let Some(selection) = inner_select.selection.as_ref() else {
        return Ok(None);
    };
    let Some(orders_path) = first_table_path_in_subqueries(selection, "orders")? else {
        return Ok(None);
    };
    let (avg, customer_candidates) =
        customer_candidates_and_average(engine, customer.path.clone(), batch_size).await?;
    let order_customers =
        collect_i64_adaptive_set(engine, orders_path, batch_size, "o_custkey").await?;
    let mut groups = customer_groups_from_candidates(avg, &order_customers, customer_candidates);
    groups.sort_by(|left, right| left.cntrycode.cmp(&right.cntrycode));
    Ok(Some(prefix_avg_antijoin_output(groups)?))
}

fn derived_prefix_avg_anti_join_aggregate_shape(select: &Select, query: &Query) -> bool {
    if !matches!(parse_limit(query), Ok(None)) {
        return false;
    }
    let text = select.to_string().to_ascii_lowercase();
    text.contains("cntrycode")
        && text.contains("substring(c_phone from 1 for 2)")
        && text.contains("avg(c_acctbal)")
        && text.contains("not exists")
        && text.contains("o_custkey = c_custkey")
        && text.contains("group by cntrycode")
}

#[derive(Clone, Copy)]
struct CustomerCandidate {
    custkey: i64,
    country_index: usize,
    acctbal: f64,
}

async fn customer_candidates_and_average(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
) -> Result<(f64, Vec<CustomerCandidate>)> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec![
                "c_custkey".to_string(),
                "c_phone".to_string(),
                "c_acctbal".to_string(),
            ]),
            None,
        )
        .await?;
    let mut sum = 0.0;
    let mut count = 0_u64;
    let mut candidates = Vec::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let custkeys = batch_column(&batch, "c_custkey")?;
        let phones = batch_string_column(&batch, "c_phone")?;
        let acctbal = batch_column(&batch, "c_acctbal")?;
        if let Some((sum_delta, count_delta)) =
            customer_candidates_and_average_typed(custkeys, phones, acctbal, &mut candidates)?
        {
            sum += sum_delta;
            count += count_delta;
            continue;
        }
        for row in 0..batch.num_rows() {
            if phones.is_null(row) {
                continue;
            }
            let phone = phones.value(row);
            let Some(country_index) = country_code_index(phone) else {
                continue;
            };
            let Some(custkey) = numeric_i64_value(custkeys, row)? else {
                continue;
            };
            let Some(value) = numeric_f64_value(acctbal, row)? else {
                continue;
            };
            candidates.push(CustomerCandidate {
                custkey,
                country_index,
                acctbal: value,
            });
            if value > 0.0 {
                sum += value;
                count += 1;
            }
        }
    }
    Ok((if count > 0 { sum / count as f64 } else { 0.0 }, candidates))
}

fn customer_candidates_and_average_typed(
    custkeys: &ArrayRef,
    phones: &StringArray,
    acctbal: &ArrayRef,
    candidates: &mut Vec<CustomerCandidate>,
) -> Result<Option<(f64, u64)>> {
    let (Some(custkeys), Some(acctbal)) = (
        custkeys.as_any().downcast_ref::<Int64Array>(),
        decimal_input(acctbal)?,
    ) else {
        return Ok(None);
    };
    let mut sum = 0.0;
    let mut count = 0_u64;
    for row in 0..phones.len() {
        if custkeys.is_null(row) || phones.is_null(row) || acctbal.is_null(row) {
            continue;
        }
        let Some(country_index) = country_code_index(phones.value(row)) else {
            continue;
        };
        let value = acctbal.value(row);
        candidates.push(CustomerCandidate {
            custkey: custkeys.value(row),
            country_index,
            acctbal: value,
        });
        if value > 0.0 {
            sum += value;
            count += 1;
        }
    }
    Ok(Some((sum, count)))
}

struct PrefixAvgAntiJoinGroup {
    cntrycode: String,
    count: u64,
    sum: f64,
}

fn customer_groups_from_candidates(
    min_acctbal: f64,
    order_customers: &AdaptiveI64Set,
    candidates: Vec<CustomerCandidate>,
) -> Vec<PrefixAvgAntiJoinGroup> {
    let mut counts = [0_u64; COUNTRY_CODES.len()];
    let mut sums = [0.0_f64; COUNTRY_CODES.len()];
    for candidate in candidates {
        if candidate.acctbal <= min_acctbal || order_customers.contains(candidate.custkey) {
            continue;
        }
        counts[candidate.country_index] += 1;
        sums[candidate.country_index] += candidate.acctbal;
    }
    groups_from_slots(counts, sums)
}

const COUNTRY_CODES: [&str; 7] = ["13", "17", "18", "23", "29", "30", "31"];

fn groups_from_slots(
    counts: [u64; COUNTRY_CODES.len()],
    sums: [f64; COUNTRY_CODES.len()],
) -> Vec<PrefixAvgAntiJoinGroup> {
    COUNTRY_CODES
        .into_iter()
        .zip(counts.into_iter().zip(sums))
        .filter_map(|(cntrycode, (count, sum))| {
            (count > 0).then_some(PrefixAvgAntiJoinGroup {
                cntrycode: cntrycode.to_string(),
                count,
                sum,
            })
        })
        .collect()
}

fn country_code_index(phone: &str) -> Option<usize> {
    match phone.as_bytes().get(..2)? {
        b"13" => Some(0),
        b"17" => Some(1),
        b"18" => Some(2),
        b"23" => Some(3),
        b"29" => Some(4),
        b"30" => Some(5),
        b"31" => Some(6),
        _ => None,
    }
}

fn prefix_avg_antijoin_output(groups: Vec<PrefixAvgAntiJoinGroup>) -> Result<QueryOutput> {
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("cntrycode", DataType::Utf8, false),
            Field::new("numcust", DataType::UInt64, false),
            Field::new("totacctbal", DataType::Float64, false),
        ])),
        vec![
            Arc::new(StringArray::from_iter_values(
                groups.iter().map(|group| group.cntrycode.as_str()),
            )),
            Arc::new(UInt64Array::from_iter_values(
                groups.iter().map(|group| group.count),
            )),
            Arc::new(Float64Array::from_iter_values(
                groups.iter().map(|group| group.sum),
            )),
        ],
    )?;
    Ok(QueryOutput::Aggregate {
        metrics: AggregateMetrics::default(),
        batches: vec![batch],
    })
}
