use super::*;

pub(super) fn join_aggregate_chunk_size() -> usize {
    std::env::var("DODAM_JOIN_AGG_CHUNK_SIZE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(4)
}

pub(super) fn build_map_chunk_size() -> usize {
    std::env::var("DODAM_BUILD_MAP_CHUNK_SIZE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(4)
}

pub(super) fn rule_chunk_size(default_chunk: usize) -> usize {
    std::env::var("DODAM_RULE_CHUNK_SIZE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default_chunk)
}

pub(super) fn scan_aggregate_fusion_enabled() -> bool {
    std::env::var("DODAM_DISABLE_SCAN_AGG_FUSION")
        .map(|value| !matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(true)
}

pub(super) fn scan_aggregate_row_group_chunk() -> usize {
    std::env::var("DODAM_SCAN_AGG_ROW_GROUP_CHUNK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(4)
}

pub(super) fn row_group_map_chunk_size(query_env: &str, default_chunk: usize) -> usize {
    let requested_chunk = std::env::var("DODAM_ROW_GROUP_MAP_CHUNK")
        .or_else(|_| std::env::var(query_env))
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0);
    choose_row_group_map_chunk(RowGroupMapChunkCostInput {
        requested_chunk,
        default_chunk,
    })
}

pub(super) fn generic_row_group_map_chunk_size(default_chunk: usize) -> usize {
    let requested_chunk = std::env::var("DODAM_ROW_GROUP_MAP_CHUNK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0);
    choose_row_group_map_chunk(RowGroupMapChunkCostInput {
        requested_chunk,
        default_chunk,
    })
}

pub(super) fn parallel_batch_fold<Partial, Output, Map, Merge>(
    stream: &mut SendableBatchStream,
    map: Map,
    mut output: Output,
    mut merge: Merge,
    label: &str,
) -> Result<Output>
where
    Partial: Send + 'static,
    Map: Fn(RecordBatch) -> Result<Partial> + Send + Sync + Clone + 'static,
    Merge: FnMut(&mut Output, Partial),
{
    let profile = tpch_profile_enabled();
    let started = profile.then(Instant::now);
    let (sender, receiver) = mpsc::channel();
    let mut pending_batches = 0_usize;
    let stream_started = profile.then(Instant::now);
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let sender = sender.clone();
        let map = map.clone();
        pending_batches += 1;
        rayon::spawn(move || {
            let _ = sender.send(map(batch));
        });
    }
    let stream_ms = stream_started
        .map(|started| started.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or_default();
    drop(sender);
    let merge_started = profile.then(Instant::now);
    for _ in 0..pending_batches {
        let partial = receiver
            .recv()
            .map_err(|_| DodamError::UnsupportedSql(format!("{label} worker stopped")))??;
        merge(&mut output, partial);
    }
    if let Some(started) = started {
        let merge_ms = merge_started
            .map(|started| started.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or_default();
        eprintln!(
            "[dodam:tpch-profile] {label}: total={:.3} ms stream_read={:.3} ms worker_wait_merge={:.3} ms batches={pending_batches}",
            started.elapsed().as_secs_f64() * 1000.0,
            stream_ms,
            merge_ms
        );
    }
    Ok(output)
}

pub(super) fn parallel_batch_fold_view_chunks<
    State,
    Partial,
    Output,
    Build,
    Consume,
    Finish,
    Merge,
>(
    stream: &mut SendableBatchStream,
    chunk_size: usize,
    build_state: Build,
    consume_batch: Consume,
    finish: Finish,
    mut output: Output,
    mut merge: Merge,
    label: &str,
) -> Result<Output>
where
    State: Send + 'static,
    Partial: Send + 'static,
    Build: Fn() -> State + Send + Sync + Clone + 'static,
    Consume: for<'a> FnMut(BatchView<'a>, &mut State) -> Result<Option<()>>
        + Clone
        + Send
        + Sync
        + 'static,
    Finish: Fn(State) -> Result<Partial> + Send + Sync + Clone + 'static,
    Merge: FnMut(&mut Output, Partial),
{
    let profile = tpch_profile_enabled();
    let started = profile.then(Instant::now);
    let (sender, receiver) = mpsc::channel();
    let mut pending_chunks = 0_usize;
    let chunk_size = chunk_size.max(1);
    let mut chunk = Vec::with_capacity(chunk_size);
    let stream_started = profile.then(Instant::now);
    while let Some(batch) = stream.next() {
        chunk.push(batch?);
        if chunk.len() < chunk_size {
            continue;
        }
        let sender = sender.clone();
        let build_state = build_state.clone();
        let mut consume_batch = consume_batch.clone();
        let finish = finish.clone();
        let task_chunk = std::mem::replace(&mut chunk, Vec::with_capacity(chunk_size));
        pending_chunks += 1;
        rayon::spawn(move || {
            let result = (|| {
                let mut state = build_state();
                for batch in &task_chunk {
                    if consume_batch(BatchView::new(batch), &mut state)?.is_none() {
                        return finish(state);
                    }
                }
                finish(state)
            })();
            let _ = sender.send(result);
        });
    }
    if !chunk.is_empty() {
        let sender = sender.clone();
        let build_state = build_state.clone();
        let mut consume_batch = consume_batch.clone();
        let finish = finish.clone();
        pending_chunks += 1;
        rayon::spawn(move || {
            let result = (|| {
                let mut state = build_state();
                for batch in &chunk {
                    if consume_batch(BatchView::new(batch), &mut state)?.is_none() {
                        return finish(state);
                    }
                }
                finish(state)
            })();
            let _ = sender.send(result);
        });
    }
    let stream_ms = stream_started
        .map(|started| started.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or_default();
    drop(sender);
    let merge_started = profile.then(Instant::now);
    for _ in 0..pending_chunks {
        let partial = receiver
            .recv()
            .map_err(|_| DodamError::UnsupportedSql(format!("{label} worker stopped")))??;
        merge(&mut output, partial);
    }
    if let Some(started) = started {
        let merge_ms = merge_started
            .map(|started| started.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or_default();
        eprintln!(
            "[dodam:tpch-profile] {label}: total={:.3} ms stream_read={:.3} ms worker_wait_merge={:.3} ms view_chunks={pending_chunks}",
            started.elapsed().as_secs_f64() * 1000.0,
            stream_ms,
            merge_ms
        );
    }
    Ok(output)
}

pub(super) fn parallel_batch_collect_pairs_view_chunks<K, V, Consume>(
    stream: &mut SendableBatchStream,
    chunk_size: usize,
    consume_batch: Consume,
    label: &str,
) -> Result<Vec<(K, V)>>
where
    K: Send + 'static,
    V: Send + 'static,
    Consume: for<'a> FnMut(BatchView<'a>, &mut Vec<(K, V)>) -> Result<Option<()>>
        + Clone
        + Send
        + Sync
        + 'static,
{
    let profile = tpch_profile_enabled();
    let started = profile.then(Instant::now);
    let (sender, receiver) = mpsc::channel();
    let mut pending_chunks = 0_usize;
    let chunk_size = chunk_size.max(1);
    let mut chunk = Vec::with_capacity(chunk_size);
    let stream_started = profile.then(Instant::now);
    while let Some(batch) = stream.next() {
        chunk.push(batch?);
        if chunk.len() < chunk_size {
            continue;
        }
        let sender = sender.clone();
        let mut consume_batch = consume_batch.clone();
        let task_chunk = std::mem::replace(&mut chunk, Vec::with_capacity(chunk_size));
        pending_chunks += 1;
        rayon::spawn(move || {
            let result = (|| {
                let mut pairs = Vec::<(K, V)>::new();
                for batch in &task_chunk {
                    if consume_batch(BatchView::new(batch), &mut pairs)?.is_none() {
                        return Ok::<Vec<(K, V)>, DodamError>(pairs);
                    }
                }
                Ok::<Vec<(K, V)>, DodamError>(pairs)
            })();
            let _ = sender.send(result);
        });
    }
    if !chunk.is_empty() {
        let sender = sender.clone();
        let mut consume_batch = consume_batch.clone();
        pending_chunks += 1;
        rayon::spawn(move || {
            let result = (|| {
                let mut pairs = Vec::<(K, V)>::new();
                for batch in &chunk {
                    if consume_batch(BatchView::new(batch), &mut pairs)?.is_none() {
                        return Ok::<Vec<(K, V)>, DodamError>(pairs);
                    }
                }
                Ok::<Vec<(K, V)>, DodamError>(pairs)
            })();
            let _ = sender.send(result);
        });
    }
    let stream_ms = stream_started
        .map(|started| started.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or_default();
    drop(sender);

    let wait_started = profile.then(Instant::now);
    let mut partials = Vec::with_capacity(pending_chunks);
    let mut total_pairs = 0_usize;
    let mut max_partial_pairs = 0_usize;
    for _ in 0..pending_chunks {
        let partial = receiver
            .recv()
            .map_err(|_| DodamError::UnsupportedSql(format!("{label} worker stopped")))??;
        total_pairs = total_pairs.saturating_add(partial.len());
        max_partial_pairs = max_partial_pairs.max(partial.len());
        partials.push(partial);
    }
    let wait_ms = wait_started
        .map(|started| started.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or_default();

    let flatten_started = profile.then(Instant::now);
    let mut output = Vec::with_capacity(total_pairs);
    for partial in partials {
        output.extend(partial);
    }
    if let Some(started) = started {
        let flatten_ms = flatten_started
            .map(|started| started.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or_default();
        eprintln!(
            "[dodam:tpch-profile] {label}: total={:.3} ms stream_read={:.3} ms worker_wait_merge={:.3} ms worker_wait={:.3} ms flatten={:.3} ms view_chunks={pending_chunks} pairs={total_pairs} max_partial_pairs={max_partial_pairs}",
            started.elapsed().as_secs_f64() * 1000.0,
            stream_ms,
            wait_ms + flatten_ms,
            wait_ms,
            flatten_ms,
        );
        eprintln!(
            "[dodam:physical] kind=pair_collect status=parallel_pair_collect total_ms={:.3} stream_read_ms={:.3} worker_wait_ms={:.3} flatten_ms={:.3} chunks={pending_chunks} pairs={total_pairs} max_partial_pairs={max_partial_pairs} label={label}",
            started.elapsed().as_secs_f64() * 1000.0,
            stream_ms,
            wait_ms,
            flatten_ms,
        );
    }
    Ok(output)
}

pub(super) fn fast_hash_map_from_pairs_profiled<K, V>(
    pairs: Vec<(K, V)>,
    label: &str,
) -> FastHashMap<K, V>
where
    K: Eq + std::hash::Hash,
{
    let profile = tpch_profile_enabled();
    let pair_count = pairs.len();
    let started = profile.then(Instant::now);
    let output = fast_hash_map_from_pairs_inner(pairs);
    if let Some(started) = started {
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "[dodam:tpch-profile] {label}: hash_map_build={elapsed_ms:.3} ms pairs={pair_count} entries={}",
            output.len()
        );
        eprintln!(
            "[dodam:physical] kind=build_lookup status=hash_map_from_pairs build_ms={elapsed_ms:.3} pairs={pair_count} entries={} label={label}",
            output.len()
        );
    }
    output
}

fn fast_hash_map_from_pairs_inner<K, V>(pairs: Vec<(K, V)>) -> FastHashMap<K, V>
where
    K: Eq + std::hash::Hash,
{
    let mut output = fast_hash_map_with_capacity(pairs.len());
    for (key, value) in pairs {
        output.insert(key, value);
    }
    output
}

pub(super) async fn collect_i64_utf8_prefix_bool_lookup(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    key_column: &str,
    string_column: &str,
    prefix: &str,
) -> Result<DenseI64BoolLookup> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec![key_column.to_string(), string_column.to_string()]),
            None,
        )
        .await?;
    let mut lookup = DenseI64BoolLookup::default();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        collect_i64_utf8_prefix_bool_lookup_batch(
            &batch,
            key_column,
            string_column,
            prefix,
            &mut lookup,
        )?;
    }
    Ok(lookup)
}

pub(super) async fn collect_i64_adaptive_set(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    key_column: &str,
) -> Result<AdaptiveI64Set> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec![key_column.to_string()]),
            None,
        )
        .await?;
    let mut keys = AdaptiveI64Set::new_dense();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        collect_i64_adaptive_set_batch(&batch, key_column, &mut keys)?;
    }
    Ok(keys)
}

pub(super) async fn collect_i64_by_i64_set_adaptive_set(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    output_key_column: &str,
    filter_key_column: &str,
    filter_keys: &HashSet<i64>,
) -> Result<AdaptiveI64Set> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec![
                output_key_column.to_string(),
                filter_key_column.to_string(),
            ]),
            None,
        )
        .await?;
    let mut output = AdaptiveI64Set::new_dense();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        collect_i64_by_i64_set_adaptive_set_batch(
            &batch,
            output_key_column,
            filter_key_column,
            filter_keys,
            &mut output,
        )?;
    }
    Ok(output)
}

pub(super) async fn collect_i64_by_i64_set_hash_map(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    output_key_column: &str,
    filter_key_column: &str,
    filter_keys: &AdaptiveI64Set,
) -> Result<FastHashMap<i64, i64>> {
    if let Some(pairs) = collect_i64_i64_mapped_pairs_direct(
        engine,
        &path,
        batch_size,
        output_key_column,
        filter_key_column,
        &|filter_key| filter_keys.contains(filter_key).then_some(filter_key),
    )? {
        return Ok(fast_hash_map_from_pairs_profiled(
            pairs,
            "direct i64-by-i64 set map build",
        ));
    }

    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec![
                output_key_column.to_string(),
                filter_key_column.to_string(),
            ]),
            None,
        )
        .await?;
    let mut output = fast_hash_map::<i64, i64>();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        collect_i64_by_i64_set_hash_map_batch(
            &batch,
            output_key_column,
            filter_key_column,
            filter_keys,
            &mut output,
        )?;
    }
    Ok(output)
}

pub(super) async fn collect_i64_by_i64_mapped_hash_map<V, Map>(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    output_key_column: &str,
    lookup_key_column: &str,
    map_value: Map,
) -> Result<FastHashMap<i64, V>>
where
    V: Copy + Send,
    Map: Fn(i64) -> Option<V> + Sync,
{
    if let Some(pairs) = collect_i64_i64_mapped_pairs_direct(
        engine,
        &path,
        batch_size,
        output_key_column,
        lookup_key_column,
        &map_value,
    )? {
        return Ok(fast_hash_map_from_pairs_profiled(
            pairs,
            "direct i64-by-i64 mapped map build",
        ));
    }

    let mut map_value = map_value;
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec![
                output_key_column.to_string(),
                lookup_key_column.to_string(),
            ]),
            None,
        )
        .await?;
    let mut output = fast_hash_map::<i64, V>();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        collect_i64_by_i64_mapped_hash_map_batch(
            &batch,
            output_key_column,
            lookup_key_column,
            &mut map_value,
            &mut output,
        )?;
    }
    Ok(output)
}

fn collect_i64_i64_mapped_pairs_direct<V, Map>(
    engine: &DodamEngine,
    path: &Path,
    batch_size: usize,
    output_key_column: &str,
    lookup_key_column: &str,
    map_value: &Map,
) -> Result<Option<Vec<(i64, V)>>>
where
    V: Copy + Send,
    Map: Fn(i64) -> Option<V> + Sync,
{
    let row_groups = (0..engine.parquet_row_group_count(path)?).collect::<Vec<_>>();
    let Some((pairs, _metrics)) = engine.scan_parquet_primitive_columns_parallel_view_fold(
        path.to_path_buf(),
        batch_size,
        row_groups,
        vec![
            (
                output_key_column.to_string(),
                DirectPrimitiveColumnType::I64,
            ),
            (
                lookup_key_column.to_string(),
                DirectPrimitiveColumnType::I64,
            ),
        ],
        Vec::<(i64, V)>::new,
        |pairs, view| {
            let (Some(output_keys), Some(lookup_keys)) = (view.i64_vector(0), view.i64_vector(1))
            else {
                return Err(DodamError::UnsupportedSql(
                    "direct i64 pair collector vector shape mismatch".to_string(),
                ));
            };
            if output_keys.len() != lookup_keys.len() {
                return Err(DodamError::UnsupportedSql(
                    "direct i64 pair collector length mismatch".to_string(),
                ));
            }
            if let (Some(output_keys), Some(lookup_keys)) = (
                output_keys.values_if_null_free(),
                lookup_keys.values_if_null_free(),
            ) {
                for block_start in (0..output_keys.len()).step_by(64) {
                    let block_end = (block_start + 64).min(output_keys.len());
                    for row in block_start..block_end {
                        if let Some(value) = map_value(lookup_keys[row]) {
                            pairs.push((output_keys[row], value));
                        }
                    }
                }
                return Ok(());
            }
            for block_start in (0..output_keys.len()).step_by(64) {
                let block_end = (block_start + 64).min(output_keys.len());
                for row in block_start..block_end {
                    if output_keys.is_null(row) || lookup_keys.is_null(row) {
                        continue;
                    }
                    if let Some(value) = map_value(lookup_keys.value(row)) {
                        pairs.push((output_keys.value(row), value));
                    }
                }
            }
            Ok(())
        },
        |pairs, mut partial| {
            pairs.append(&mut partial);
            Ok(())
        },
    )?
    else {
        return Ok(None);
    };
    Ok(Some(pairs))
}

pub(super) async fn collect_i64_i64_adaptive_map(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    key_column: &str,
    value_column: &str,
) -> Result<AdaptiveI64Map<i64>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec![key_column.to_string(), value_column.to_string()]),
            None,
        )
        .await?;
    let mut output = AdaptiveI64Map::<i64>::new_dense();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        collect_i64_i64_adaptive_map_batch(&batch, key_column, value_column, &mut output)?;
    }
    Ok(output)
}

pub(super) async fn collect_i64_utf8_prefix_set(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    key_column: &str,
    string_column: &str,
    prefix: &str,
) -> Result<HashSet<i64>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec![key_column.to_string(), string_column.to_string()]),
            None,
        )
        .await?;
    let mut keys = HashSet::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        collect_i64_utf8_prefix_set_batch(&batch, key_column, string_column, prefix, &mut keys)?;
    }
    Ok(keys)
}

pub(super) async fn collect_i64_utf8_eq_set(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    key_column: &str,
    string_column: &str,
    expected: &str,
) -> Result<HashSet<i64>> {
    let row_groups = (0..engine.parquet_row_group_count(&path)?).collect::<Vec<_>>();
    if let Some((keys, _metrics)) = engine
        .collect_parquet_i64_by_utf8_dictionary_predicate_parallel(
            path.clone(),
            row_groups,
            key_column.to_string(),
            string_column.to_string(),
            (1, 4),
            |value| value == expected.as_bytes(),
        )?
    {
        return Ok(keys.into_iter().collect());
    }

    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec![key_column.to_string(), string_column.to_string()]),
            None,
        )
        .await?;
    let mut keys = HashSet::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        collect_i64_utf8_eq_set_batch(&batch, key_column, string_column, expected, &mut keys)?;
    }
    Ok(keys)
}

pub(super) async fn collect_i64_utf8_eq_adaptive_set(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    key_column: &str,
    string_column: &str,
    expected: &str,
) -> Result<AdaptiveI64Set> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec![key_column.to_string(), string_column.to_string()]),
            None,
        )
        .await?;
    let mut keys = AdaptiveI64Set::new_dense();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        collect_i64_utf8_eq_adaptive_set_batch(
            &batch,
            key_column,
            string_column,
            expected.as_bytes(),
            &mut keys,
        )?;
    }
    Ok(keys)
}

pub(super) async fn collect_i64_two_utf8_eq_set(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    key_column: &str,
    first_string_column: &str,
    first_expected: &str,
    second_string_column: &str,
    second_expected: &str,
) -> Result<HashSet<i64>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec![
                key_column.to_string(),
                first_string_column.to_string(),
                second_string_column.to_string(),
            ]),
            None,
        )
        .await?;
    let mut keys = HashSet::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        collect_i64_two_utf8_eq_set_batch(
            &batch,
            key_column,
            first_string_column,
            first_expected,
            second_string_column,
            second_expected,
            &mut keys,
        )?;
    }
    Ok(keys)
}

pub(super) async fn collect_i64_two_utf8_i64_mapped_adaptive_map<V, MapRaw, MapFallback>(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    key_column: &str,
    first_string_column: &str,
    second_string_column: &str,
    numeric_column: &str,
    map_raw: MapRaw,
    map_fallback: MapFallback,
) -> Result<AdaptiveI64Map<V>>
where
    V: Copy + Default,
    MapRaw: Fn(&[u8], &[u8], i64) -> Option<V>,
    MapFallback: Fn(&str, &str, f64) -> Option<V>,
{
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec![
                key_column.to_string(),
                first_string_column.to_string(),
                second_string_column.to_string(),
                numeric_column.to_string(),
            ]),
            None,
        )
        .await?;
    let mut output = AdaptiveI64Map::<V>::new_dense();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        collect_i64_two_utf8_i64_mapped_adaptive_map_batch(
            &batch,
            key_column,
            first_string_column,
            second_string_column,
            numeric_column,
            &map_raw,
            &map_fallback,
            &mut output,
        )?;
    }
    Ok(output)
}

pub(super) async fn collect_i64_i64_i64_mapped_set<MapRaw, MapFallback>(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    first_column: &str,
    second_column: &str,
    third_column: &str,
    map_raw: MapRaw,
    map_fallback: MapFallback,
) -> Result<HashSet<i64>>
where
    MapRaw: Fn(i64, i64, i64) -> Option<i64>,
    MapFallback: Fn(i64, i64, f64) -> Option<i64>,
{
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec![
                first_column.to_string(),
                second_column.to_string(),
                third_column.to_string(),
            ]),
            None,
        )
        .await?;
    let mut output = HashSet::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        collect_i64_i64_i64_mapped_set_batch(
            &batch,
            first_column,
            second_column,
            third_column,
            &map_raw,
            &map_fallback,
            &mut output,
        )?;
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn collect_i64_three_utf8_mapped_rows_pruned<T, Map>(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    key_column: &str,
    first_string_column: &str,
    second_string_column: &str,
    third_string_column: &str,
    pruning_predicates: Vec<Expr>,
    map: Map,
) -> Result<Vec<T>>
where
    Map: Fn(i64, &str, &str, &str) -> Option<T>,
{
    let projection = Projection::Columns(vec![
        key_column.to_string(),
        first_string_column.to_string(),
        second_string_column.to_string(),
        third_string_column.to_string(),
    ]);
    let mut stream = if pruning_predicates.is_empty() {
        engine
            .scan_parquet_batches(path, batch_size, None, projection, None)
            .await?
    } else {
        engine
            .scan_parquet_batches_pruned(path, batch_size, projection, pruning_predicates)
            .await?
    };
    let mut output = Vec::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        collect_i64_three_utf8_mapped_rows_batch(
            &batch,
            key_column,
            first_string_column,
            second_string_column,
            third_string_column,
            &map,
            &mut output,
        )?;
    }
    Ok(output)
}

pub(super) async fn collect_i64_i64_two_utf8_mapped_rows<T, Map>(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    first_key_column: &str,
    second_key_column: &str,
    first_string_column: &str,
    second_string_column: &str,
    map: Map,
) -> Result<Vec<T>>
where
    Map: Fn(i64, i64, &str, &str) -> Option<T>,
{
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec![
                first_key_column.to_string(),
                second_key_column.to_string(),
                first_string_column.to_string(),
                second_string_column.to_string(),
            ]),
            None,
        )
        .await?;
    let mut output = Vec::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        collect_i64_i64_two_utf8_mapped_rows_batch(
            &batch,
            first_key_column,
            second_key_column,
            first_string_column,
            second_string_column,
            &map,
            &mut output,
        )?;
    }
    Ok(output)
}

pub(super) async fn collect_i64_i64_utf8_mapped_hash_map<V, Map>(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    output_key_column: &str,
    filter_key_column: &str,
    string_column: &str,
    map: Map,
) -> Result<HashMap<i64, V>>
where
    Map: Fn(i64, i64, &str) -> Option<V>,
{
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec![
                output_key_column.to_string(),
                filter_key_column.to_string(),
                string_column.to_string(),
            ]),
            None,
        )
        .await?;
    let mut output = HashMap::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        collect_i64_i64_utf8_mapped_hash_map_batch(
            &batch,
            output_key_column,
            filter_key_column,
            string_column,
            &map,
            &mut output,
        )?;
    }
    Ok(output)
}

pub(super) async fn collect_i64_utf8_contains_set(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    key_column: &str,
    string_column: &str,
    needle: &str,
) -> Result<HashSet<i64>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec![key_column.to_string(), string_column.to_string()]),
            None,
        )
        .await?;
    let key_column = key_column.to_string();
    let string_column = string_column.to_string();
    let needle = needle.to_string();
    parallel_batch_fold(
        &mut stream,
        move |batch| {
            collect_i64_utf8_contains_set_batch(batch, &key_column, &string_column, &needle)
        },
        HashSet::<i64>::new(),
        merge_sets,
        "i64 utf8 contains set",
    )
}

fn collect_i64_utf8_prefix_bool_lookup_batch(
    batch: &RecordBatch,
    key_column: &str,
    string_column: &str,
    prefix: &str,
    lookup: &mut DenseI64BoolLookup,
) -> Result<()> {
    let keys = batch_column(batch, key_column)?;
    let strings = batch_string_column(batch, string_column)?;
    if let Some(keys) = keys.as_any().downcast_ref::<Int64Array>()
        && keys.null_count() == 0
    {
        for row in 0..batch.num_rows() {
            if strings.is_valid(row) {
                lookup.insert(keys.value(row), strings.value(row).starts_with(prefix));
            }
        }
        return Ok(());
    }
    for row in 0..batch.num_rows() {
        if strings.is_null(row) {
            continue;
        }
        if let Some(key) = numeric_i64_value(keys, row)? {
            lookup.insert(key, strings.value(row).starts_with(prefix));
        }
    }
    Ok(())
}

fn collect_i64_adaptive_set_batch(
    batch: &RecordBatch,
    key_column: &str,
    output: &mut AdaptiveI64Set,
) -> Result<()> {
    let keys = batch_column(batch, key_column)?;
    if let Some(keys) = keys.as_any().downcast_ref::<Int64Array>() {
        if keys.null_count() == 0 {
            if output.try_insert_dense_values(keys.values().as_ref()) {
                return Ok(());
            }
            for &key in keys.values() {
                output.insert(key);
            }
            return Ok(());
        }
        for row in 0..keys.len() {
            if keys.is_valid(row) {
                output.insert(keys.value(row));
            }
        }
        return Ok(());
    }
    for row in 0..batch.num_rows() {
        if let Some(key) = numeric_i64_value(keys, row)? {
            output.insert(key);
        }
    }
    Ok(())
}

fn collect_i64_by_i64_set_adaptive_set_batch(
    batch: &RecordBatch,
    output_key_column: &str,
    filter_key_column: &str,
    filter_keys: &HashSet<i64>,
    output: &mut AdaptiveI64Set,
) -> Result<()> {
    let output_keys = batch_column(batch, output_key_column)?;
    let filter_values = batch_column(batch, filter_key_column)?;
    if let (Some(output_keys), Some(filter_values)) = (
        output_keys.as_any().downcast_ref::<Int64Array>(),
        filter_values.as_any().downcast_ref::<Int64Array>(),
    ) {
        if output_keys.null_count() == 0 && filter_values.null_count() == 0 {
            for (&output_key, &filter_key) in
                output_keys.values().iter().zip(filter_values.values())
            {
                if filter_keys.contains(&filter_key) {
                    output.insert(output_key);
                }
            }
            return Ok(());
        }
        for row in 0..batch.num_rows() {
            if output_keys.is_valid(row)
                && filter_values.is_valid(row)
                && filter_keys.contains(&filter_values.value(row))
            {
                output.insert(output_keys.value(row));
            }
        }
        return Ok(());
    }
    for row in 0..batch.num_rows() {
        let (Some(output_key), Some(filter_key)) = (
            numeric_i64_value(output_keys, row)?,
            numeric_i64_value(filter_values, row)?,
        ) else {
            continue;
        };
        if filter_keys.contains(&filter_key) {
            output.insert(output_key);
        }
    }
    Ok(())
}

fn collect_i64_by_i64_set_hash_map_batch(
    batch: &RecordBatch,
    output_key_column: &str,
    filter_key_column: &str,
    filter_keys: &AdaptiveI64Set,
    output: &mut FastHashMap<i64, i64>,
) -> Result<()> {
    let output_keys = batch_column(batch, output_key_column)?;
    let filter_values = batch_column(batch, filter_key_column)?;
    if let (Some(output_keys), Some(filter_values)) = (
        output_keys.as_any().downcast_ref::<Int64Array>(),
        filter_values.as_any().downcast_ref::<Int64Array>(),
    ) {
        let dense_filter = filter_keys.dense_contains_slice();
        if output_keys.null_count() == 0 && filter_values.null_count() == 0 {
            let output_values = output_keys.values().as_ref();
            let filter_values = filter_values.values().as_ref();
            for (&output_key, &filter_key) in output_values.iter().zip(filter_values) {
                if filter_keys.contains_cached(dense_filter, filter_key) {
                    output.insert(output_key, filter_key);
                }
            }
            return Ok(());
        }
        for row in 0..batch.num_rows() {
            if output_keys.is_valid(row) && filter_values.is_valid(row) {
                let filter_key = filter_values.value(row);
                if filter_keys.contains_cached(dense_filter, filter_key) {
                    output.insert(output_keys.value(row), filter_key);
                }
            }
        }
        return Ok(());
    }
    for row in 0..batch.num_rows() {
        let (Some(output_key), Some(filter_key)) = (
            numeric_i64_value(output_keys, row)?,
            numeric_i64_value(filter_values, row)?,
        ) else {
            continue;
        };
        if filter_keys.contains(filter_key) {
            output.insert(output_key, filter_key);
        }
    }
    Ok(())
}

fn collect_i64_by_i64_mapped_hash_map_batch<V, Map>(
    batch: &RecordBatch,
    output_key_column: &str,
    lookup_key_column: &str,
    map_value: &mut Map,
    output: &mut FastHashMap<i64, V>,
) -> Result<()>
where
    V: Copy,
    Map: FnMut(i64) -> Option<V>,
{
    let output_keys = batch_column(batch, output_key_column)?;
    let lookup_values = batch_column(batch, lookup_key_column)?;
    if let (Some(output_keys), Some(lookup_values)) = (
        output_keys.as_any().downcast_ref::<Int64Array>(),
        lookup_values.as_any().downcast_ref::<Int64Array>(),
    ) {
        if output_keys.null_count() == 0 && lookup_values.null_count() == 0 {
            let output_values = output_keys.values().as_ref();
            let lookup_values = lookup_values.values().as_ref();
            for (&output_key, &lookup_key) in output_values.iter().zip(lookup_values) {
                if let Some(value) = map_value(lookup_key) {
                    output.insert(output_key, value);
                }
            }
            return Ok(());
        }
        for row in 0..batch.num_rows() {
            if output_keys.is_valid(row) && lookup_values.is_valid(row) {
                let lookup_key = lookup_values.value(row);
                if let Some(value) = map_value(lookup_key) {
                    output.insert(output_keys.value(row), value);
                }
            }
        }
        return Ok(());
    }
    for row in 0..batch.num_rows() {
        let (Some(output_key), Some(lookup_key)) = (
            numeric_i64_value(output_keys, row)?,
            numeric_i64_value(lookup_values, row)?,
        ) else {
            continue;
        };
        if let Some(value) = map_value(lookup_key) {
            output.insert(output_key, value);
        }
    }
    Ok(())
}

fn collect_i64_i64_adaptive_map_batch(
    batch: &RecordBatch,
    key_column: &str,
    value_column: &str,
    output: &mut AdaptiveI64Map<i64>,
) -> Result<()> {
    let keys = batch_column(batch, key_column)?;
    let values = batch_column(batch, value_column)?;
    if let (Some(keys), Some(values)) = (
        keys.as_any().downcast_ref::<Int64Array>(),
        values.as_any().downcast_ref::<Int64Array>(),
    ) {
        if keys.null_count() == 0 && values.null_count() == 0 {
            let key_values = keys.values().as_ref();
            let value_values = values.values().as_ref();
            for (&key, &value) in key_values.iter().zip(value_values) {
                output.insert(key, value);
            }
            return Ok(());
        }
        for row in 0..batch.num_rows() {
            if keys.is_valid(row) && values.is_valid(row) {
                output.insert(keys.value(row), values.value(row));
            }
        }
        return Ok(());
    }
    for row in 0..batch.num_rows() {
        let (Some(key), Some(value)) = (
            numeric_i64_value(keys, row)?,
            numeric_i64_value(values, row)?,
        ) else {
            continue;
        };
        output.insert(key, value);
    }
    Ok(())
}

fn collect_i64_utf8_prefix_set_batch(
    batch: &RecordBatch,
    key_column: &str,
    string_column: &str,
    prefix: &str,
    output: &mut HashSet<i64>,
) -> Result<()> {
    let keys = batch_column(batch, key_column)?;
    let strings = batch_string_column(batch, string_column)?;
    if let Some(keys) = keys.as_any().downcast_ref::<Int64Array>()
        && keys.null_count() == 0
    {
        for row in 0..batch.num_rows() {
            if strings.is_valid(row) && strings.value(row).starts_with(prefix) {
                output.insert(keys.value(row));
            }
        }
        return Ok(());
    }
    for row in 0..batch.num_rows() {
        if strings.is_null(row) || !strings.value(row).starts_with(prefix) {
            continue;
        }
        if let Some(key) = numeric_i64_value(keys, row)? {
            output.insert(key);
        }
    }
    Ok(())
}

fn collect_i64_utf8_eq_set_batch(
    batch: &RecordBatch,
    key_column: &str,
    string_column: &str,
    expected: &str,
    output: &mut HashSet<i64>,
) -> Result<()> {
    let keys = batch_column(batch, key_column)?;
    let strings = batch_string_column(batch, string_column)?;
    if let Some(keys) = keys.as_any().downcast_ref::<Int64Array>()
        && keys.null_count() == 0
    {
        for row in 0..batch.num_rows() {
            if strings.is_valid(row) && strings.value(row) == expected {
                output.insert(keys.value(row));
            }
        }
        return Ok(());
    }
    for row in 0..batch.num_rows() {
        if strings.is_null(row) || strings.value(row) != expected {
            continue;
        }
        if let Some(key) = numeric_i64_value(keys, row)? {
            output.insert(key);
        }
    }
    Ok(())
}

fn collect_i64_utf8_eq_adaptive_set_batch(
    batch: &RecordBatch,
    key_column: &str,
    string_column: &str,
    expected: &[u8],
    output: &mut AdaptiveI64Set,
) -> Result<()> {
    let keys = batch_column(batch, key_column)?;
    let strings = batch_string_column(batch, string_column)?;
    if let Some(keys) = keys.as_any().downcast_ref::<Int64Array>() {
        if keys.null_count() == 0 && strings.null_count() == 0 {
            let key_values = keys.values().as_ref();
            let offsets = strings.value_offsets();
            let values = strings.value_data();
            for row in 0..key_values.len() {
                let start = offsets[row] as usize;
                let end = offsets[row + 1] as usize;
                if &values[start..end] == expected {
                    output.insert(key_values[row]);
                }
            }
            return Ok(());
        }
        for row in 0..keys.len() {
            if keys.is_valid(row)
                && strings.is_valid(row)
                && strings.value(row).as_bytes() == expected
            {
                output.insert(keys.value(row));
            }
        }
        return Ok(());
    }
    let expected = std::str::from_utf8(expected)
        .map_err(|error| DodamError::UnsupportedSql(error.to_string()))?;
    for row in 0..batch.num_rows() {
        if strings.is_null(row) || strings.value(row) != expected {
            continue;
        }
        if let Some(key) = numeric_i64_value(keys, row)? {
            output.insert(key);
        }
    }
    Ok(())
}

fn collect_i64_two_utf8_eq_set_batch(
    batch: &RecordBatch,
    key_column: &str,
    first_string_column: &str,
    first_expected: &str,
    second_string_column: &str,
    second_expected: &str,
    output: &mut HashSet<i64>,
) -> Result<()> {
    let keys = batch_column(batch, key_column)?;
    let first_strings = batch_string_column(batch, first_string_column)?;
    let second_strings = batch_string_column(batch, second_string_column)?;
    if let Some(keys) = keys.as_any().downcast_ref::<Int64Array>()
        && keys.null_count() == 0
    {
        for row in 0..batch.num_rows() {
            if first_strings.is_valid(row)
                && second_strings.is_valid(row)
                && first_strings.value(row) == first_expected
                && second_strings.value(row) == second_expected
            {
                output.insert(keys.value(row));
            }
        }
        return Ok(());
    }
    for row in 0..batch.num_rows() {
        if first_strings.is_null(row)
            || second_strings.is_null(row)
            || first_strings.value(row) != first_expected
            || second_strings.value(row) != second_expected
        {
            continue;
        }
        if let Some(key) = numeric_i64_value(keys, row)? {
            output.insert(key);
        }
    }
    Ok(())
}

fn collect_i64_two_utf8_i64_mapped_adaptive_map_batch<V, MapRaw, MapFallback>(
    batch: &RecordBatch,
    key_column: &str,
    first_string_column: &str,
    second_string_column: &str,
    numeric_column: &str,
    map_raw: &MapRaw,
    map_fallback: &MapFallback,
    output: &mut AdaptiveI64Map<V>,
) -> Result<()>
where
    V: Copy + Default,
    MapRaw: Fn(&[u8], &[u8], i64) -> Option<V>,
    MapFallback: Fn(&str, &str, f64) -> Option<V>,
{
    let keys = batch_column(batch, key_column)?;
    let first_strings = batch_string_column(batch, first_string_column)?;
    let second_strings = batch_string_column(batch, second_string_column)?;
    let numbers = batch_column(batch, numeric_column)?;
    if let (Some(keys), Some(numbers)) = (
        keys.as_any().downcast_ref::<Int64Array>(),
        i64_or_i32_values(numbers),
    ) && keys.null_count() == 0
        && first_strings.null_count() == 0
        && second_strings.null_count() == 0
        && numbers.null_count() == 0
    {
        let first_offsets = first_strings.value_offsets();
        let first_data = first_strings.value_data();
        let second_offsets = second_strings.value_offsets();
        let second_data = second_strings.value_data();
        for row in 0..batch.num_rows() {
            let first = bytes_string_parts(first_offsets, first_data, row);
            let second = bytes_string_parts(second_offsets, second_data, row);
            if let Some(value) = map_raw(first, second, numbers.value(row)) {
                output.insert(keys.value(row), value);
            }
        }
        return Ok(());
    }
    for row in 0..batch.num_rows() {
        if first_strings.is_null(row) || second_strings.is_null(row) {
            continue;
        }
        let (Some(key), Some(number)) = (
            numeric_i64_value(keys, row)?,
            numeric_f64_value(numbers, row)?,
        ) else {
            continue;
        };
        if let Some(value) =
            map_fallback(first_strings.value(row), second_strings.value(row), number)
        {
            output.insert(key, value);
        }
    }
    Ok(())
}

fn collect_i64_i64_i64_mapped_set_batch<MapRaw, MapFallback>(
    batch: &RecordBatch,
    first_column: &str,
    second_column: &str,
    third_column: &str,
    map_raw: &MapRaw,
    map_fallback: &MapFallback,
    output: &mut HashSet<i64>,
) -> Result<()>
where
    MapRaw: Fn(i64, i64, i64) -> Option<i64>,
    MapFallback: Fn(i64, i64, f64) -> Option<i64>,
{
    let first = batch_column(batch, first_column)?;
    let second = batch_column(batch, second_column)?;
    let third = batch_column(batch, third_column)?;
    if let (Some(first), Some(second), Some(third)) = (
        first.as_any().downcast_ref::<Int64Array>(),
        second.as_any().downcast_ref::<Int64Array>(),
        i64_or_i32_values(third),
    ) && first.null_count() == 0
        && second.null_count() == 0
        && third.null_count() == 0
    {
        let first_values = first.values().as_ref();
        let second_values = second.values().as_ref();
        match third {
            I64OrI32Values::I32(values) => {
                for ((&first, &second), &third) in first_values
                    .iter()
                    .zip(second_values)
                    .zip(values.values().as_ref())
                {
                    if let Some(value) = map_raw(first, second, i64::from(third)) {
                        output.insert(value);
                    }
                }
            }
            I64OrI32Values::I64(values) => {
                for ((&first, &second), &third) in first_values
                    .iter()
                    .zip(second_values)
                    .zip(values.values().as_ref())
                {
                    if let Some(value) = map_raw(first, second, third) {
                        output.insert(value);
                    }
                }
            }
        }
        return Ok(());
    }
    for row in 0..batch.num_rows() {
        let (Some(first), Some(second), Some(third)) = (
            numeric_i64_value(first, row)?,
            numeric_i64_value(second, row)?,
            numeric_f64_value(third, row)?,
        ) else {
            continue;
        };
        if let Some(value) = map_fallback(first, second, third) {
            output.insert(value);
        }
    }
    Ok(())
}

fn collect_i64_three_utf8_mapped_rows_batch<T, Map>(
    batch: &RecordBatch,
    key_column: &str,
    first_string_column: &str,
    second_string_column: &str,
    third_string_column: &str,
    map: &Map,
    output: &mut Vec<T>,
) -> Result<()>
where
    Map: Fn(i64, &str, &str, &str) -> Option<T>,
{
    let keys = batch_column(batch, key_column)?;
    let first_strings = batch_string_column(batch, first_string_column)?;
    let second_strings = batch_string_column(batch, second_string_column)?;
    let third_strings = batch_string_column(batch, third_string_column)?;
    if let Some(keys) = keys.as_any().downcast_ref::<Int64Array>() {
        for row in 0..batch.num_rows() {
            if keys.is_null(row)
                || first_strings.is_null(row)
                || second_strings.is_null(row)
                || third_strings.is_null(row)
            {
                continue;
            }
            if let Some(value) = map(
                keys.value(row),
                first_strings.value(row),
                second_strings.value(row),
                third_strings.value(row),
            ) {
                output.push(value);
            }
        }
        return Ok(());
    }
    for row in 0..batch.num_rows() {
        if first_strings.is_null(row) || second_strings.is_null(row) || third_strings.is_null(row) {
            continue;
        }
        let Some(key) = numeric_i64_value(keys, row)? else {
            continue;
        };
        if let Some(value) = map(
            key,
            first_strings.value(row),
            second_strings.value(row),
            third_strings.value(row),
        ) {
            output.push(value);
        }
    }
    Ok(())
}

fn collect_i64_i64_two_utf8_mapped_rows_batch<T, Map>(
    batch: &RecordBatch,
    first_key_column: &str,
    second_key_column: &str,
    first_string_column: &str,
    second_string_column: &str,
    map: &Map,
    output: &mut Vec<T>,
) -> Result<()>
where
    Map: Fn(i64, i64, &str, &str) -> Option<T>,
{
    let first_keys = batch_column(batch, first_key_column)?;
    let second_keys = batch_column(batch, second_key_column)?;
    let first_strings = batch_string_column(batch, first_string_column)?;
    let second_strings = batch_string_column(batch, second_string_column)?;
    if let (Some(first_keys), Some(second_keys)) = (
        first_keys.as_any().downcast_ref::<Int64Array>(),
        second_keys.as_any().downcast_ref::<Int64Array>(),
    ) {
        for row in 0..batch.num_rows() {
            if first_keys.is_null(row)
                || second_keys.is_null(row)
                || first_strings.is_null(row)
                || second_strings.is_null(row)
            {
                continue;
            }
            if let Some(value) = map(
                first_keys.value(row),
                second_keys.value(row),
                first_strings.value(row),
                second_strings.value(row),
            ) {
                output.push(value);
            }
        }
        return Ok(());
    }
    for row in 0..batch.num_rows() {
        if first_strings.is_null(row) || second_strings.is_null(row) {
            continue;
        }
        let (Some(first_key), Some(second_key)) = (
            numeric_i64_value(first_keys, row)?,
            numeric_i64_value(second_keys, row)?,
        ) else {
            continue;
        };
        if let Some(value) = map(
            first_key,
            second_key,
            first_strings.value(row),
            second_strings.value(row),
        ) {
            output.push(value);
        }
    }
    Ok(())
}

fn collect_i64_i64_utf8_mapped_hash_map_batch<V, Map>(
    batch: &RecordBatch,
    output_key_column: &str,
    filter_key_column: &str,
    string_column: &str,
    map: &Map,
    output: &mut HashMap<i64, V>,
) -> Result<()>
where
    Map: Fn(i64, i64, &str) -> Option<V>,
{
    let output_keys = batch_column(batch, output_key_column)?;
    let filter_keys = batch_column(batch, filter_key_column)?;
    let strings = batch_string_column(batch, string_column)?;
    if let (Some(output_keys), Some(filter_keys)) = (
        output_keys.as_any().downcast_ref::<Int64Array>(),
        filter_keys.as_any().downcast_ref::<Int64Array>(),
    ) {
        for row in 0..batch.num_rows() {
            if output_keys.is_null(row) || filter_keys.is_null(row) || strings.is_null(row) {
                continue;
            }
            if let Some(value) = map(
                output_keys.value(row),
                filter_keys.value(row),
                strings.value(row),
            ) {
                output.insert(output_keys.value(row), value);
            }
        }
        return Ok(());
    }
    for row in 0..batch.num_rows() {
        if strings.is_null(row) {
            continue;
        }
        let (Some(output_key), Some(filter_key)) = (
            numeric_i64_value(output_keys, row)?,
            numeric_i64_value(filter_keys, row)?,
        ) else {
            continue;
        };
        if let Some(value) = map(output_key, filter_key, strings.value(row)) {
            output.insert(output_key, value);
        }
    }
    Ok(())
}

enum I64OrI32Values<'a> {
    I32(&'a Int32Array),
    I64(&'a Int64Array),
}

impl I64OrI32Values<'_> {
    fn null_count(&self) -> usize {
        match self {
            Self::I32(values) => values.null_count(),
            Self::I64(values) => values.null_count(),
        }
    }

    fn value(&self, row: usize) -> i64 {
        match self {
            Self::I32(values) => i64::from(values.value(row)),
            Self::I64(values) => values.value(row),
        }
    }
}

fn i64_or_i32_values(column: &ArrayRef) -> Option<I64OrI32Values<'_>> {
    column
        .as_any()
        .downcast_ref::<Int32Array>()
        .map(I64OrI32Values::I32)
        .or_else(|| {
            column
                .as_any()
                .downcast_ref::<Int64Array>()
                .map(I64OrI32Values::I64)
        })
}

fn collect_i64_utf8_contains_set_batch(
    batch: RecordBatch,
    key_column: &str,
    string_column: &str,
    needle: &str,
) -> Result<HashSet<i64>> {
    let keys = batch_column(&batch, key_column)?;
    let strings = batch_string_column(&batch, string_column)?;
    let finder = Finder::new(needle.as_bytes());
    let mut output = HashSet::new();
    if let Some(keys) = keys.as_any().downcast_ref::<Int64Array>()
        && keys.null_count() == 0
    {
        for row in 0..batch.num_rows() {
            if strings.is_valid(row) && finder.find(strings.value(row).as_bytes()).is_some() {
                output.insert(keys.value(row));
            }
        }
        return Ok(output);
    }
    for row in 0..batch.num_rows() {
        if strings.is_null(row) || finder.find(strings.value(row).as_bytes()).is_none() {
            continue;
        }
        if let Some(key) = numeric_i64_value(keys, row)? {
            output.insert(key);
        }
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn discounted_revenue_by_i64_bool_lookup_batch(
    batch: RecordBatch,
    key_column: &str,
    date_column: &str,
    extendedprice_column: &str,
    discount_column: &str,
    start_days: i32,
    end_days: i32,
    lookup: &DenseI64BoolLookup,
) -> Result<(f64, f64)> {
    let keys = batch_column(&batch, key_column)?;
    let dates = batch_column(&batch, date_column)?;
    let extendedprices = batch_column(&batch, extendedprice_column)?;
    let discounts = batch_column(&batch, discount_column)?;
    let mut matched = 0.0;
    let mut total = 0.0;
    if let (Some(keys), Some(dates), Some(extendedprices), Some(discounts)) = (
        keys.as_any().downcast_ref::<Int64Array>(),
        dates.as_any().downcast_ref::<Date32Array>(),
        decimal_input(extendedprices)?,
        decimal_input(discounts)?,
    ) {
        for row in 0..batch.num_rows() {
            if keys.is_null(row)
                || dates.is_null(row)
                || extendedprices.is_null(row)
                || discounts.is_null(row)
            {
                continue;
            }
            let date = dates.value(row);
            if date < start_days || date >= end_days {
                continue;
            }
            let Some(is_matched) = lookup.get(keys.value(row)) else {
                continue;
            };
            let value = extendedprices.value(row) * (1.0 - discounts.value(row));
            if is_matched {
                matched += value;
            }
            total += value;
        }
        return Ok((matched, total));
    }
    for row in 0..batch.num_rows() {
        let Some(date) = date32_value(dates, row)? else {
            continue;
        };
        if date < start_days || date >= end_days {
            continue;
        }
        let (Some(key), Some(extendedprice), Some(discount)) = (
            numeric_i64_value(keys, row)?,
            numeric_f64_value(extendedprices, row)?,
            numeric_f64_value(discounts, row)?,
        ) else {
            continue;
        };
        let Some(is_matched) = lookup.get(key) else {
            continue;
        };
        let value = extendedprice * (1.0 - discount);
        if is_matched {
            matched += value;
        }
        total += value;
    }
    Ok((matched, total))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn update_i64_grouped_discounted_revenue_by_date_view<S: BuildHasher>(
    view: BatchView<'_>,
    key_index: usize,
    date_index: usize,
    extendedprice_index: usize,
    discount_index: usize,
    start_days: i32,
    end_days: i32,
    revenues: &mut HashMap<i64, f64, S>,
) -> Result<bool> {
    let (Some(keys), Some(dates), Some(extendedprices), Some(discounts)) = (
        view.i64_vector(key_index),
        view.date32_vector(date_index),
        view.decimal128_vector(extendedprice_index),
        view.decimal128_vector(discount_index),
    ) else {
        return Ok(false);
    };
    let (Some(key_values), Some(date_values)) =
        (keys.values_if_null_free(), dates.values_if_null_free())
    else {
        return Ok(false);
    };
    consume_filtered_discounted_revenue_decimal128_vectors(
        extendedprices,
        discounts,
        view.num_rows(),
        |row| {
            let date = date_values[row];
            Ok(date >= start_days && date < end_days)
        },
        |row, revenue| {
            *revenues.entry(key_values[row]).or_insert(0.0) += revenue;
            Ok(())
        },
    )?;
    Ok(true)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn try_direct_i64_grouped_discounted_revenue_by_date(
    engine: &DodamEngine,
    path: &Path,
    batch_size: usize,
    key_column: &str,
    date_column: &str,
    extendedprice_column: &str,
    discount_column: &str,
    start_days: i32,
    end_days: i32,
) -> Result<Option<HashMap<i64, f64>>> {
    let Some(key_types) =
        engine.parquet_direct_primitive_column_types(path, &[key_column.to_string()])?
    else {
        return Ok(None);
    };
    if !matches!(key_types.as_slice(), [DirectPrimitiveColumnType::I64])
        || !engine.parquet_is_date32_column(path, date_column)?
    {
        return Ok(None);
    }
    let Some((extendedprice_precision, extendedprice_scale)) =
        engine.parquet_decimal128_type(path, extendedprice_column)?
    else {
        return Ok(None);
    };
    let Some((discount_precision, discount_scale)) =
        engine.parquet_decimal128_type(path, discount_column)?
    else {
        return Ok(None);
    };
    if extendedprice_precision > 18 || discount_precision > 18 {
        return Ok(None);
    }

    if let Some(revenues) = try_direct_dense_atomic_i64_grouped_discounted_revenue_by_date(
        engine,
        path,
        batch_size,
        key_column,
        date_column,
        extendedprice_column,
        discount_column,
        extendedprice_precision,
        extendedprice_scale,
        discount_precision,
        discount_scale,
        start_days,
        end_days,
    )? {
        return Ok(Some(revenues));
    }

    let row_groups = (0..engine.parquet_row_group_count(path)?).collect::<Vec<_>>();
    let Some((revenues, _metrics)) = engine.scan_parquet_primitive_columns_parallel_view_fold(
        path.to_path_buf(),
        batch_size,
        row_groups,
        vec![
            (key_column.to_string(), DirectPrimitiveColumnType::I64),
            (date_column.to_string(), DirectPrimitiveColumnType::Date32),
            (
                extendedprice_column.to_string(),
                DirectPrimitiveColumnType::Decimal128Int64Raw {
                    precision: extendedprice_precision,
                    scale: extendedprice_scale,
                },
            ),
            (
                discount_column.to_string(),
                DirectPrimitiveColumnType::Decimal128Int64Raw {
                    precision: discount_precision,
                    scale: discount_scale,
                },
            ),
        ],
        HashMap::<i64, f64>::new,
        move |revenues, view| {
            if update_i64_grouped_discounted_revenue_by_date_view(
                view, 0, 1, 2, 3, start_days, end_days, revenues,
            )? {
                Ok(())
            } else {
                Err(DodamError::UnsupportedSql(
                    "direct grouped discounted-revenue vector shape mismatch".to_string(),
                ))
            }
        },
        |revenues, partial| {
            merge_f64_groups(revenues, partial);
            Ok(())
        },
    )?
    else {
        return Ok(None);
    };
    Ok(Some(revenues))
}

struct DenseAtomicI64GroupedSum {
    sums: Box<[AtomicI64]>,
    present: Box<[AtomicU8]>,
}

impl DenseAtomicI64GroupedSum {
    fn new(max_key: usize) -> Self {
        Self {
            sums: (0..=max_key)
                .map(|_| AtomicI64::new(0))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            present: (0..=max_key)
                .map(|_| AtomicU8::new(0))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        }
    }

    #[inline(always)]
    fn add(&self, key: usize, value: i64) {
        self.sums[key].fetch_add(value, Ordering::Relaxed);
        self.present[key].store(1, Ordering::Relaxed);
    }

    fn to_hash_map(&self, revenue_scale: f64) -> HashMap<i64, f64> {
        self.sums
            .iter()
            .zip(self.present.iter())
            .enumerate()
            .filter_map(|(key, (sum, present))| {
                (present.load(Ordering::Relaxed) != 0).then(|| {
                    (
                        key as i64,
                        sum.load(Ordering::Relaxed) as f64 * revenue_scale,
                    )
                })
            })
            .collect()
    }
}

#[allow(clippy::too_many_arguments)]
fn try_direct_dense_atomic_i64_grouped_discounted_revenue_by_date(
    engine: &DodamEngine,
    path: &Path,
    batch_size: usize,
    key_column: &str,
    date_column: &str,
    extendedprice_column: &str,
    discount_column: &str,
    extendedprice_precision: u8,
    extendedprice_scale: i8,
    discount_precision: u8,
    discount_scale: i8,
    start_days: i32,
    end_days: i32,
) -> Result<Option<HashMap<i64, f64>>> {
    let Some(key_ranges) =
        engine.parquet_primitive_column_min_max_by_row_group(path, key_column)?
    else {
        return Ok(None);
    };
    let Some(date_ranges) =
        engine.parquet_primitive_column_min_max_by_row_group(path, date_column)?
    else {
        return Ok(None);
    };
    let Some(extendedprice_ranges) =
        engine.parquet_primitive_column_min_max_by_row_group(path, extendedprice_column)?
    else {
        return Ok(None);
    };
    let Some(discount_ranges) =
        engine.parquet_primitive_column_min_max_by_row_group(path, discount_column)?
    else {
        return Ok(None);
    };
    if [
        &key_ranges,
        &date_ranges,
        &extendedprice_ranges,
        &discount_ranges,
    ]
    .into_iter()
    .flatten()
    .any(|range| range.null_count != Some(0))
    {
        return Ok(None);
    }

    let Some(min_key) = key_ranges.iter().map(|range| range.min).min() else {
        return Ok(Some(HashMap::new()));
    };
    let Some(max_key) = key_ranges.iter().map(|range| range.max).max() else {
        return Ok(Some(HashMap::new()));
    };
    if min_key < 0 {
        return Ok(None);
    }
    let Ok(max_key) = i64::try_from(max_key) else {
        return Ok(None);
    };
    let Some(max_key) = adaptive_dense_index(max_key, DEFAULT_MAX_DENSE_I64_KEY) else {
        return Ok(None);
    };
    let Some(discount_factor) = decimal_raw_scale_i64(discount_scale) else {
        return Ok(None);
    };
    if !dense_atomic_discounted_revenue_sum_fits(
        &key_ranges,
        &extendedprice_ranges,
        &discount_ranges,
        discount_factor,
    ) {
        return Ok(None);
    }
    let revenue_scale = 10_f64.powi(-i32::from(extendedprice_scale) - i32::from(discount_scale));
    let row_groups = (0..key_ranges.len()).collect::<Vec<_>>();
    let selected_sums = Arc::new(DenseAtomicI64GroupedSum::new(max_key));
    let shared_sums = selected_sums.clone();
    if engine
        .scan_parquet_dictionary_date_range_selected_primitive_columns_parallel(
            path.to_path_buf(),
            row_groups.clone(),
            date_column.to_string(),
            start_days,
            end_days,
            vec![
                (key_column.to_string(), DirectPrimitiveColumnType::I64),
                (
                    extendedprice_column.to_string(),
                    DirectPrimitiveColumnType::Decimal128Int64Raw {
                        precision: extendedprice_precision,
                        scale: extendedprice_scale,
                    },
                ),
                (
                    discount_column.to_string(),
                    DirectPrimitiveColumnType::Decimal128Int64Raw {
                        precision: discount_precision,
                        scale: discount_scale,
                    },
                ),
            ],
            move |view| {
                update_dense_atomic_i64_grouped_discounted_revenue_selected_view(
                    view,
                    &shared_sums,
                    max_key,
                    discount_factor,
                )
            },
        )?
        .is_some()
    {
        return Ok(Some(selected_sums.to_hash_map(revenue_scale)));
    }

    let sums = Arc::new(DenseAtomicI64GroupedSum::new(max_key));
    let shared_sums = sums.clone();
    let Some((_state, _metrics)) = engine.scan_parquet_primitive_columns_parallel_view_fold(
        path.to_path_buf(),
        batch_size,
        row_groups,
        vec![
            (key_column.to_string(), DirectPrimitiveColumnType::I64),
            (date_column.to_string(), DirectPrimitiveColumnType::Date32),
            (
                extendedprice_column.to_string(),
                DirectPrimitiveColumnType::Decimal128Int64Raw {
                    precision: extendedprice_precision,
                    scale: extendedprice_scale,
                },
            ),
            (
                discount_column.to_string(),
                DirectPrimitiveColumnType::Decimal128Int64Raw {
                    precision: discount_precision,
                    scale: discount_scale,
                },
            ),
        ],
        move || shared_sums.clone(),
        move |sums, view| {
            update_dense_atomic_i64_grouped_discounted_revenue_by_date_view(
                view,
                sums,
                max_key,
                discount_factor,
                start_days,
                end_days,
            )
        },
        |_sums, _partial| Ok(()),
    )?
    else {
        return Ok(None);
    };
    Ok(Some(sums.to_hash_map(revenue_scale)))
}

fn decimal_raw_scale_i64(scale: i8) -> Option<i64> {
    let scale = u32::try_from(scale).ok()?;
    10_i64.checked_pow(scale)
}

fn dense_atomic_discounted_revenue_sum_fits(
    key_ranges: &[PrimitiveRowGroupMinMax],
    extendedprice_ranges: &[PrimitiveRowGroupMinMax],
    discount_ranges: &[PrimitiveRowGroupMinMax],
    discount_factor: i64,
) -> bool {
    let max_extendedprice = extendedprice_ranges
        .iter()
        .flat_map(|range| [range.min, range.max])
        .map(i128::abs)
        .max()
        .unwrap_or_default();
    let max_discount_factor = discount_ranges
        .iter()
        .flat_map(|range| [range.min, range.max])
        .map(|discount| (i128::from(discount_factor) - discount).abs())
        .max()
        .unwrap_or_default();
    let rows = key_ranges.iter().try_fold(0_i128, |rows, range| {
        rows.checked_add(i128::try_from(range.rows).ok()?)
    });
    max_extendedprice
        .checked_mul(max_discount_factor)
        .and_then(|value| value.checked_mul(rows?))
        .is_some_and(|value| value <= i128::from(i64::MAX))
}

#[allow(clippy::too_many_arguments)]
fn update_dense_atomic_i64_grouped_discounted_revenue_by_date_view(
    view: BatchView<'_>,
    sums: &DenseAtomicI64GroupedSum,
    max_key: usize,
    discount_factor: i64,
    start_days: i32,
    end_days: i32,
) -> Result<()> {
    let (Some(keys), Some(dates), Some(extendedprices), Some(discounts)) = (
        view.i64_vector(0),
        view.date32_vector(1),
        view.decimal128_vector(2),
        view.decimal128_vector(3),
    ) else {
        return Err(DodamError::UnsupportedSql(
            "direct dense atomic grouped discounted-revenue vector shape mismatch".to_string(),
        ));
    };
    let (Some(keys), Some(dates), Some(extendedprices), Some(discounts)) = (
        keys.values_if_null_free(),
        dates.values_if_null_free(),
        extendedprices.raw_i64_values(),
        discounts.raw_i64_values(),
    ) else {
        return Err(DodamError::UnsupportedSql(
            "direct dense atomic grouped discounted-revenue requires null-free raw vectors"
                .to_string(),
        ));
    };
    for row in 0..view.num_rows() {
        let date = dates[row];
        if date < start_days || date >= end_days {
            continue;
        }
        let key = usize::try_from(keys[row]).map_err(|_| {
            DodamError::UnsupportedSql(
                "direct dense atomic grouped discounted-revenue key is negative".to_string(),
            )
        })?;
        if key > max_key {
            return Err(DodamError::UnsupportedSql(
                "direct dense atomic grouped discounted-revenue key exceeds metadata range"
                    .to_string(),
            ));
        }
        let value = extendedprices[row]
            .checked_mul(discount_factor - discounts[row])
            .ok_or_else(|| {
                DodamError::UnsupportedSql(
                    "direct dense atomic grouped discounted-revenue value overflow".to_string(),
                )
            })?;
        sums.add(key, value);
    }
    Ok(())
}

fn update_dense_atomic_i64_grouped_discounted_revenue_selected_view(
    view: BatchView<'_>,
    sums: &DenseAtomicI64GroupedSum,
    max_key: usize,
    discount_factor: i64,
) -> Result<()> {
    let (Some(keys), Some(extendedprices), Some(discounts)) = (
        view.i64_vector(0),
        view.decimal128_vector(1),
        view.decimal128_vector(2),
    ) else {
        return Err(DodamError::UnsupportedSql(
            "selected dictionary grouped discounted-revenue vector shape mismatch".to_string(),
        ));
    };
    let (Some(keys), Some(extendedprices), Some(discounts)) = (
        keys.values_if_null_free(),
        extendedprices.raw_i64_values(),
        discounts.raw_i64_values(),
    ) else {
        return Err(DodamError::UnsupportedSql(
            "selected dictionary grouped discounted-revenue requires null-free raw vectors"
                .to_string(),
        ));
    };
    for ((&key, &extendedprice), &discount) in
        keys.iter().zip(extendedprices.iter()).zip(discounts.iter())
    {
        let key = usize::try_from(key).map_err(|_| {
            DodamError::UnsupportedSql(
                "selected dictionary grouped discounted-revenue key is negative".to_string(),
            )
        })?;
        if key > max_key {
            return Err(DodamError::UnsupportedSql(
                "selected dictionary grouped discounted-revenue key exceeds metadata range"
                    .to_string(),
            ));
        }
        let value = extendedprice
            .checked_mul(discount_factor - discount)
            .ok_or_else(|| {
                DodamError::UnsupportedSql(
                    "selected dictionary grouped discounted-revenue value overflow".to_string(),
                )
            })?;
        sums.add(key, value);
    }
    Ok(())
}

pub(super) fn consume_discounted_revenue_decimal128_vectors<Consume>(
    extendedprices: Decimal128VectorView<'_>,
    discounts: Decimal128VectorView<'_>,
    row_count: usize,
    mut consume: Consume,
) -> Result<()>
where
    Consume: FnMut(usize, Option<f64>) -> Result<()>,
{
    if extendedprices.null_count() == 0 && discounts.null_count() == 0 {
        let discount_scale = discounts.scale();
        let revenue_scale = 1.0 / (extendedprices.scale() * discount_scale);
        if let (Some(extendedprice_values), Some(discount_values)) =
            (extendedprices.raw_i64_values(), discounts.raw_i64_values())
        {
            for row in 0..row_count {
                consume(
                    row,
                    Some(decimal_discounted_revenue_raw_i64(
                        extendedprice_values[row],
                        discount_values[row],
                        discount_scale,
                        revenue_scale,
                    )),
                )?;
            }
            return Ok(());
        }
        if extendedprices.raw_i64_values().is_none()
            && discounts.raw_i64_values().is_none()
            && extendedprices.raw_i64_bytes().is_none()
            && discounts.raw_i64_bytes().is_none()
        {
            let extendedprice_values = extendedprices.raw_values();
            let discount_values = discounts.raw_values();
            for row in 0..row_count {
                consume(
                    row,
                    Some(decimal_discounted_revenue_raw(
                        extendedprice_values[row],
                        discount_values[row],
                        discount_scale,
                        revenue_scale,
                    )),
                )?;
            }
            return Ok(());
        }
        for row in 0..row_count {
            consume(
                row,
                Some(extendedprices.value(row) * (1.0 - discounts.value(row))),
            )?;
        }
        return Ok(());
    }
    for row in 0..row_count {
        if extendedprices.is_null(row) || discounts.is_null(row) {
            consume(row, None)?;
            continue;
        }
        consume(
            row,
            Some(extendedprices.value(row) * (1.0 - discounts.value(row))),
        )?;
    }
    Ok(())
}

#[inline]
pub(super) fn discounted_revenue_minus_product(
    extendedprice: f64,
    discount: f64,
    product_left: f64,
    product_right: f64,
) -> f64 {
    extendedprice * (1.0 - discount) - product_left * product_right
}

pub(super) fn consume_filtered_discounted_revenue_decimal128_vectors<Predicate, Consume>(
    extendedprices: Decimal128VectorView<'_>,
    discounts: Decimal128VectorView<'_>,
    row_count: usize,
    mut predicate: Predicate,
    mut consume: Consume,
) -> Result<()>
where
    Predicate: FnMut(usize) -> Result<bool>,
    Consume: FnMut(usize, f64) -> Result<()>,
{
    consume_filtered_discounted_revenue_decimal128_vectors_with_payload(
        extendedprices,
        discounts,
        row_count,
        |row| Ok(predicate(row)?.then_some(())),
        |row, (), revenue| consume(row, revenue),
    )
}

pub(super) fn consume_filtered_discounted_revenue_decimal128_vectors_with_payload<
    Payload,
    Predicate,
    Consume,
>(
    extendedprices: Decimal128VectorView<'_>,
    discounts: Decimal128VectorView<'_>,
    row_count: usize,
    mut predicate: Predicate,
    mut consume: Consume,
) -> Result<()>
where
    Predicate: FnMut(usize) -> Result<Option<Payload>>,
    Consume: FnMut(usize, Payload, f64) -> Result<()>,
{
    if extendedprices.null_count() == 0 && discounts.null_count() == 0 {
        let discount_scale = discounts.scale();
        let revenue_scale = 1.0 / (extendedprices.scale() * discount_scale);
        if let (Some(extendedprice_values), Some(discount_values)) =
            (extendedprices.raw_i64_values(), discounts.raw_i64_values())
        {
            for row in 0..row_count {
                if let Some(payload) = predicate(row)? {
                    consume(
                        row,
                        payload,
                        decimal_discounted_revenue_raw_i64(
                            extendedprice_values[row],
                            discount_values[row],
                            discount_scale,
                            revenue_scale,
                        ),
                    )?;
                }
            }
            return Ok(());
        }
        if extendedprices.raw_i64_values().is_none()
            && discounts.raw_i64_values().is_none()
            && extendedprices.raw_i64_bytes().is_none()
            && discounts.raw_i64_bytes().is_none()
        {
            let extendedprice_values = extendedprices.raw_values();
            let discount_values = discounts.raw_values();
            for row in 0..row_count {
                if let Some(payload) = predicate(row)? {
                    consume(
                        row,
                        payload,
                        decimal_discounted_revenue_raw(
                            extendedprice_values[row],
                            discount_values[row],
                            discount_scale,
                            revenue_scale,
                        ),
                    )?;
                }
            }
            return Ok(());
        }
        for row in 0..row_count {
            if let Some(payload) = predicate(row)? {
                consume(
                    row,
                    payload,
                    extendedprices.value(row) * (1.0 - discounts.value(row)),
                )?;
            }
        }
        return Ok(());
    }
    for row in 0..row_count {
        if extendedprices.is_null(row) || discounts.is_null(row) {
            continue;
        }
        if let Some(payload) = predicate(row)? {
            consume(
                row,
                payload,
                extendedprices.value(row) * (1.0 - discounts.value(row)),
            )?;
        }
    }
    Ok(())
}

pub(super) fn consume_discounted_revenue_decimal128_vectors_at_offsets<Consume>(
    extendedprices: Decimal128VectorView<'_>,
    discounts: Decimal128VectorView<'_>,
    row_count: usize,
    row_base: usize,
    row_offsets: &[u32],
    mut consume: Consume,
) -> Result<()>
where
    Consume: FnMut(usize, Option<f64>) -> Result<()>,
{
    if extendedprices.len() < row_count || discounts.len() < row_count {
        return Err(DodamError::UnsupportedSql(
            "discounted revenue vector row count mismatch".to_string(),
        ));
    }
    let discount_scale = discounts.scale();
    let revenue_scale = 1.0 / (extendedprices.scale() * discount_scale);
    if extendedprices.null_count() == 0 && discounts.null_count() == 0 {
        if let (Some(extendedprice_values), Some(discount_values)) =
            (extendedprices.raw_i64_values(), discounts.raw_i64_values())
        {
            for &row in row_offsets {
                let row = (row as usize).checked_sub(row_base).ok_or_else(|| {
                    DodamError::UnsupportedSql(
                        "discounted revenue selected row out of range".to_string(),
                    )
                })?;
                if row >= row_count {
                    return Err(DodamError::UnsupportedSql(
                        "discounted revenue selected row out of range".to_string(),
                    ));
                }
                consume(
                    row,
                    Some(decimal_discounted_revenue_raw_i64(
                        extendedprice_values[row],
                        discount_values[row],
                        discount_scale,
                        revenue_scale,
                    )),
                )?;
            }
            return Ok(());
        }
        if extendedprices.raw_i64_values().is_none()
            && discounts.raw_i64_values().is_none()
            && extendedprices.raw_i64_bytes().is_none()
            && discounts.raw_i64_bytes().is_none()
        {
            let extendedprice_values = extendedprices.raw_values();
            let discount_values = discounts.raw_values();
            for &row in row_offsets {
                let row = (row as usize).checked_sub(row_base).ok_or_else(|| {
                    DodamError::UnsupportedSql(
                        "discounted revenue selected row out of range".to_string(),
                    )
                })?;
                if row >= row_count {
                    return Err(DodamError::UnsupportedSql(
                        "discounted revenue selected row out of range".to_string(),
                    ));
                }
                consume(
                    row,
                    Some(decimal_discounted_revenue_raw(
                        extendedprice_values[row],
                        discount_values[row],
                        discount_scale,
                        revenue_scale,
                    )),
                )?;
            }
            return Ok(());
        }
    }
    for &row in row_offsets {
        let row = (row as usize).checked_sub(row_base).ok_or_else(|| {
            DodamError::UnsupportedSql("discounted revenue selected row out of range".to_string())
        })?;
        if row >= row_count {
            return Err(DodamError::UnsupportedSql(
                "discounted revenue selected row out of range".to_string(),
            ));
        }
        if extendedprices.is_null(row) || discounts.is_null(row) {
            consume(row, None)?;
        } else {
            consume(
                row,
                Some(extendedprices.value(row) * (1.0 - discounts.value(row))),
            )?;
        }
    }
    Ok(())
}

pub(super) fn collect_i64_i64_pairs_view<V, Map>(
    view: BatchView<'_>,
    unsupported_message: &str,
    map: Map,
) -> Result<Vec<(i64, V)>>
where
    Map: FnMut(i64, i64) -> Result<Option<V>>,
{
    if view.num_columns() == 2
        && let (Some(first), Some(second)) = (view.i64_vector(0), view.i64_vector(1))
    {
        return collect_i64_i64_pairs_vectors(first, second, map);
    }
    let Some(batch) = view.try_record_batch() else {
        return Err(DodamError::UnsupportedSql(unsupported_message.to_string()));
    };
    collect_i64_i64_pairs_arrays(batch.column(0), batch.column(1), map)
}

pub(super) fn collect_i64_i64_date32_pairs_view<V, Map>(
    view: BatchView<'_>,
    unsupported_message: &str,
    map: Map,
) -> Result<Vec<(i64, V)>>
where
    Map: FnMut(i64, i64, i32) -> Result<Option<V>>,
{
    if view.num_columns() == 3
        && let (Some(first), Some(second), Some(date)) = (
            view.i64_vector(0),
            view.i64_vector(1),
            view.date32_vector(2),
        )
    {
        return collect_i64_i64_date32_pairs_vectors(first, second, date, map);
    }
    let Some(batch) = view.try_record_batch() else {
        return Err(DodamError::UnsupportedSql(unsupported_message.to_string()));
    };
    collect_i64_i64_date32_pairs_arrays(batch.column(0), batch.column(1), batch.column(2), map)
}

pub(super) fn collect_i64_i64_date32_optional_i64_pairs_view<V, Map>(
    view: BatchView<'_>,
    unsupported_message: &str,
    optional_value: Option<i64>,
    map: Map,
) -> Result<Vec<(i64, V)>>
where
    Map: FnMut(i64, i64, i32, i64) -> Result<Option<V>>,
{
    if view.num_columns() == 3
        && let Some(optional_value) = optional_value
        && let (Some(first), Some(second), Some(date)) = (
            view.i64_vector(0),
            view.i64_vector(1),
            view.date32_vector(2),
        )
    {
        return collect_i64_i64_date32_const_i64_pairs_vectors(
            first,
            second,
            date,
            optional_value,
            map,
        );
    }
    if view.num_columns() == 4
        && let (Some(first), Some(second), Some(date), Some(value)) = (
            view.i64_vector(0),
            view.i64_vector(1),
            view.date32_vector(2),
            view.i64_vector(3),
        )
    {
        return collect_i64_i64_date32_i64_pairs_vectors(first, second, date, value, map);
    }
    let Some(batch) = view.try_record_batch() else {
        return Err(DodamError::UnsupportedSql(unsupported_message.to_string()));
    };
    if batch.num_columns() == 3 {
        let Some(optional_value) = optional_value else {
            return Err(DodamError::UnsupportedSql(unsupported_message.to_string()));
        };
        return collect_i64_i64_date32_const_i64_pairs_arrays(
            batch.column(0),
            batch.column(1),
            batch.column(2),
            optional_value,
            map,
        );
    }
    if batch.num_columns() == 4 {
        return collect_i64_i64_date32_i64_pairs_arrays(
            batch.column(0),
            batch.column(1),
            batch.column(2),
            batch.column(3),
            map,
        );
    }
    Err(DodamError::UnsupportedSql(unsupported_message.to_string()))
}

pub(super) fn collect_i64_i64_date32_optional_i64_pairs_view_into<V, Map>(
    view: BatchView<'_>,
    unsupported_message: &str,
    optional_value: Option<i64>,
    pairs: &mut Vec<(i64, V)>,
    mut map: Map,
) -> Result<()>
where
    Map: FnMut(i64, i64, i32, i64) -> Result<Option<V>>,
{
    if view.num_columns() == 3
        && let Some(optional_value) = optional_value
        && let (Some(first), Some(second), Some(date)) = (
            view.i64_vector(0),
            view.i64_vector(1),
            view.date32_vector(2),
        )
    {
        if let (Some(first_values), Some(second_values), Some(date_values)) = (
            first.values_if_null_free(),
            second.values_if_null_free(),
            date.values_if_null_free(),
        ) {
            for row in 0..first_values.len() {
                if let Some(mapped) = map(
                    first_values[row],
                    second_values[row],
                    date_values[row],
                    optional_value,
                )? {
                    pairs.push((first_values[row], mapped));
                }
            }
            return Ok(());
        }
        for row in 0..first.len() {
            if first.is_null(row) || second.is_null(row) || date.is_null(row) {
                continue;
            }
            let first_value = first.value(row);
            if let Some(mapped) = map(
                first_value,
                second.value(row),
                date.value(row),
                optional_value,
            )? {
                pairs.push((first_value, mapped));
            }
        }
        return Ok(());
    }
    if view.num_columns() == 4
        && let (Some(first), Some(second), Some(date), Some(value)) = (
            view.i64_vector(0),
            view.i64_vector(1),
            view.date32_vector(2),
            view.i64_vector(3),
        )
    {
        if let (Some(first_values), Some(second_values), Some(date_values), Some(values)) = (
            first.values_if_null_free(),
            second.values_if_null_free(),
            date.values_if_null_free(),
            value.values_if_null_free(),
        ) {
            for row in 0..first_values.len() {
                if let Some(mapped) = map(
                    first_values[row],
                    second_values[row],
                    date_values[row],
                    values[row],
                )? {
                    pairs.push((first_values[row], mapped));
                }
            }
            return Ok(());
        }
        for row in 0..first.len() {
            if first.is_null(row) || second.is_null(row) || date.is_null(row) || value.is_null(row)
            {
                continue;
            }
            let first_value = first.value(row);
            if let Some(mapped) = map(
                first_value,
                second.value(row),
                date.value(row),
                value.value(row),
            )? {
                pairs.push((first_value, mapped));
            }
        }
        return Ok(());
    }
    pairs.extend(collect_i64_i64_date32_optional_i64_pairs_view(
        view,
        unsupported_message,
        optional_value,
        map,
    )?);
    Ok(())
}

fn collect_i64_i64_pairs_vectors<V, Map>(
    first: I64VectorView<'_>,
    second: I64VectorView<'_>,
    mut map: Map,
) -> Result<Vec<(i64, V)>>
where
    Map: FnMut(i64, i64) -> Result<Option<V>>,
{
    let mut pairs = Vec::new();
    if let (Some(first_values), Some(second_values)) =
        (first.values_if_null_free(), second.values_if_null_free())
    {
        for row in 0..first_values.len() {
            if let Some(value) = map(first_values[row], second_values[row])? {
                pairs.push((first_values[row], value));
            }
        }
        return Ok(pairs);
    }
    for row in 0..first.len() {
        if first.is_null(row) || second.is_null(row) {
            continue;
        }
        if let Some(value) = map(first.value(row), second.value(row))? {
            pairs.push((first.value(row), value));
        }
    }
    Ok(pairs)
}

fn collect_i64_i64_date32_pairs_vectors<V, Map>(
    first: I64VectorView<'_>,
    second: I64VectorView<'_>,
    date: Date32VectorView<'_>,
    mut map: Map,
) -> Result<Vec<(i64, V)>>
where
    Map: FnMut(i64, i64, i32) -> Result<Option<V>>,
{
    let mut pairs = Vec::new();
    if let (Some(first_values), Some(second_values), Some(date_values)) = (
        first.values_if_null_free(),
        second.values_if_null_free(),
        date.values_if_null_free(),
    ) {
        for row in 0..first_values.len() {
            if let Some(value) = map(first_values[row], second_values[row], date_values[row])? {
                pairs.push((first_values[row], value));
            }
        }
        return Ok(pairs);
    }
    for row in 0..first.len() {
        if first.is_null(row) || second.is_null(row) || date.is_null(row) {
            continue;
        }
        if let Some(value) = map(first.value(row), second.value(row), date.value(row))? {
            pairs.push((first.value(row), value));
        }
    }
    Ok(pairs)
}

fn collect_i64_i64_date32_const_i64_pairs_vectors<V, Map>(
    first: I64VectorView<'_>,
    second: I64VectorView<'_>,
    date: Date32VectorView<'_>,
    value: i64,
    mut map: Map,
) -> Result<Vec<(i64, V)>>
where
    Map: FnMut(i64, i64, i32, i64) -> Result<Option<V>>,
{
    let mut pairs = Vec::new();
    if let (Some(first_values), Some(second_values), Some(date_values)) = (
        first.values_if_null_free(),
        second.values_if_null_free(),
        date.values_if_null_free(),
    ) {
        for row in 0..first_values.len() {
            if let Some(mapped) = map(
                first_values[row],
                second_values[row],
                date_values[row],
                value,
            )? {
                pairs.push((first_values[row], mapped));
            }
        }
        return Ok(pairs);
    }
    for row in 0..first.len() {
        if first.is_null(row) || second.is_null(row) || date.is_null(row) {
            continue;
        }
        if let Some(mapped) = map(first.value(row), second.value(row), date.value(row), value)? {
            pairs.push((first.value(row), mapped));
        }
    }
    Ok(pairs)
}

fn collect_i64_i64_date32_i64_pairs_vectors<V, Map>(
    first: I64VectorView<'_>,
    second: I64VectorView<'_>,
    date: Date32VectorView<'_>,
    value: I64VectorView<'_>,
    mut map: Map,
) -> Result<Vec<(i64, V)>>
where
    Map: FnMut(i64, i64, i32, i64) -> Result<Option<V>>,
{
    let mut pairs = Vec::new();
    if let (Some(first_values), Some(second_values), Some(date_values), Some(values)) = (
        first.values_if_null_free(),
        second.values_if_null_free(),
        date.values_if_null_free(),
        value.values_if_null_free(),
    ) {
        for row in 0..first_values.len() {
            if let Some(mapped) = map(
                first_values[row],
                second_values[row],
                date_values[row],
                values[row],
            )? {
                pairs.push((first_values[row], mapped));
            }
        }
        return Ok(pairs);
    }
    for row in 0..first.len() {
        if first.is_null(row) || second.is_null(row) || date.is_null(row) || value.is_null(row) {
            continue;
        }
        if let Some(mapped) = map(
            first.value(row),
            second.value(row),
            date.value(row),
            value.value(row),
        )? {
            pairs.push((first.value(row), mapped));
        }
    }
    Ok(pairs)
}

fn collect_i64_i64_pairs_arrays<V, Map>(
    first: &ArrayRef,
    second: &ArrayRef,
    mut map: Map,
) -> Result<Vec<(i64, V)>>
where
    Map: FnMut(i64, i64) -> Result<Option<V>>,
{
    let mut pairs = Vec::new();
    for row in 0..first.len() {
        let (Some(first_value), Some(second_value)) = (
            numeric_i64_value(first, row)?,
            numeric_i64_value(second, row)?,
        ) else {
            continue;
        };
        if let Some(value) = map(first_value, second_value)? {
            pairs.push((first_value, value));
        }
    }
    Ok(pairs)
}

fn collect_i64_i64_date32_const_i64_pairs_arrays<V, Map>(
    first: &ArrayRef,
    second: &ArrayRef,
    date: &ArrayRef,
    value: i64,
    mut map: Map,
) -> Result<Vec<(i64, V)>>
where
    Map: FnMut(i64, i64, i32, i64) -> Result<Option<V>>,
{
    let mut pairs = Vec::new();
    for row in 0..first.len() {
        let (Some(first_value), Some(second_value), Some(date_value)) = (
            numeric_i64_value(first, row)?,
            numeric_i64_value(second, row)?,
            date32_value(date, row)?,
        ) else {
            continue;
        };
        if let Some(mapped) = map(first_value, second_value, date_value, value)? {
            pairs.push((first_value, mapped));
        }
    }
    Ok(pairs)
}

fn collect_i64_i64_date32_i64_pairs_arrays<V, Map>(
    first: &ArrayRef,
    second: &ArrayRef,
    date: &ArrayRef,
    value: &ArrayRef,
    mut map: Map,
) -> Result<Vec<(i64, V)>>
where
    Map: FnMut(i64, i64, i32, i64) -> Result<Option<V>>,
{
    let mut pairs = Vec::new();
    for row in 0..first.len() {
        let (Some(first_value), Some(second_value), Some(date_value), Some(value)) = (
            numeric_i64_value(first, row)?,
            numeric_i64_value(second, row)?,
            date32_value(date, row)?,
            numeric_i64_value(value, row)?,
        ) else {
            continue;
        };
        if let Some(mapped) = map(first_value, second_value, date_value, value)? {
            pairs.push((first_value, mapped));
        }
    }
    Ok(pairs)
}

fn collect_i64_i64_date32_pairs_arrays<V, Map>(
    first: &ArrayRef,
    second: &ArrayRef,
    date: &ArrayRef,
    mut map: Map,
) -> Result<Vec<(i64, V)>>
where
    Map: FnMut(i64, i64, i32) -> Result<Option<V>>,
{
    let mut pairs = Vec::new();
    for row in 0..first.len() {
        let (Some(first_value), Some(second_value), Some(date_value)) = (
            numeric_i64_value(first, row)?,
            numeric_i64_value(second, row)?,
            date32_value(date, row)?,
        ) else {
            continue;
        };
        if let Some(value) = map(first_value, second_value, date_value)? {
            pairs.push((first_value, value));
        }
    }
    Ok(pairs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filtered_discounted_revenue_payload_skips_null_decimal_rows() {
        let extendedprices = Decimal128Array::from(vec![Some(10_000), None, Some(30_000)])
            .with_precision_and_scale(12, 2)
            .expect("decimal precision");
        let discounts = Decimal128Array::from(vec![Some(10), Some(20), None])
            .with_precision_and_scale(12, 2)
            .expect("decimal precision");
        let extendedprices =
            Decimal128VectorView::try_new_arrow(&extendedprices).expect("decimal view");
        let discounts = Decimal128VectorView::try_new_arrow(&discounts).expect("decimal view");
        let mut rows = Vec::new();

        consume_filtered_discounted_revenue_decimal128_vectors_with_payload(
            extendedprices,
            discounts,
            3,
            |row| Ok(Some(row)),
            |row, payload, revenue| {
                rows.push((row, payload, revenue));
                Ok(())
            },
        )
        .expect("consume discounted revenue");

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].0, 0);
        assert_eq!(rows[0].1, 0);
        assert!((rows[0].2 - 90.0).abs() < 1e-9);
    }
}
