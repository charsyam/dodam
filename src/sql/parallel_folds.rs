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
