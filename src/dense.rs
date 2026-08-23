use std::collections::{HashMap, HashSet};
use std::hash::BuildHasher;
use std::sync::atomic::{AtomicU8, Ordering};

use crate::hash::{FastHashMap, FastHashSet, fast_hash_map_with_capacity};
use rayon::prelude::*;

const DEFAULT_MAX_DENSE_I64_KEY: usize = 20_000_000;
const DEFAULT_MAX_DENSE_I64_SET_BYTES: usize = 16 * 1024 * 1024;

#[derive(Clone)]
pub(crate) enum AdaptiveI64Set {
    Dense {
        contains: Vec<bool>,
        words: Vec<u64>,
        len: usize,
        max_dense_key: usize,
    },
    Hash(FastHashSet<i64>),
}

pub(crate) struct DenseAtomicU8 {
    markers: Vec<AtomicU8>,
}

impl DenseAtomicU8 {
    pub(crate) fn zeroed(len: usize) -> Self {
        Self {
            markers: (0..len).map(|_| AtomicU8::new(0)).collect(),
        }
    }

    pub(crate) fn from_values_parallel(values: &[u8]) -> Self {
        Self {
            markers: values
                .par_iter()
                .copied()
                .map(AtomicU8::new)
                .collect::<Vec<_>>(),
        }
    }

    pub(crate) fn into_markers(self) -> Vec<AtomicU8> {
        self.markers
    }

    pub(crate) fn store_present(&self, index: usize) {
        if let Some(marker) = self.markers.get(index) {
            marker.store(1, Ordering::Relaxed);
        }
    }

    pub(crate) fn into_adaptive_i64_set(self) -> AdaptiveI64Set {
        let contains = self
            .markers
            .into_par_iter()
            .map(|marker| marker.load(Ordering::Relaxed) != 0)
            .collect::<Vec<_>>();
        let len = contains.par_iter().filter(|present| **present).count();
        let words = adaptive_i64_set_words_from_contains(&contains);
        AdaptiveI64Set::Dense {
            contains,
            words,
            len,
            max_dense_key: default_max_dense_i64_set_key(),
        }
    }
}

impl AdaptiveI64Set {
    pub(crate) fn new_dense() -> Self {
        Self::Dense {
            contains: Vec::new(),
            words: Vec::new(),
            len: 0,
            max_dense_key: default_max_dense_i64_set_key(),
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Dense { len, .. } => *len,
            Self::Hash(keys) => keys.len(),
        }
    }

    pub(crate) fn contains(&self, key: i64) -> bool {
        match self {
            Self::Dense { words, .. } => adaptive_i64_words_contains(words, key),
            Self::Hash(keys) => keys.contains(&key),
        }
    }

    pub(crate) fn dense_contains_slice(&self) -> Option<&[bool]> {
        match self {
            Self::Dense { contains, .. } => Some(contains),
            Self::Hash(_) => None,
        }
    }

    pub(crate) fn dense_word_slice(&self) -> Option<&[u64]> {
        match self {
            Self::Dense { words, .. } => Some(words),
            Self::Hash(_) => None,
        }
    }

    pub(crate) fn contains_cached(&self, dense_contains: Option<&[bool]>, key: i64) -> bool {
        if let Some(dense_contains) = dense_contains {
            return usize::try_from(key)
                .ok()
                .and_then(|index| dense_contains.get(index))
                .copied()
                .unwrap_or(false);
        }
        self.contains(key)
    }

    pub(crate) fn selective_key_range(&self) -> Option<(i64, i64)> {
        let (min_key, max_key, len) = match self {
            Self::Dense { contains, len, .. } => {
                let min_key = contains.iter().position(|contains| *contains)? as i64;
                let max_key = contains.iter().rposition(|contains| *contains)? as i64;
                (min_key, max_key, *len)
            }
            Self::Hash(keys) => {
                let min_key = keys.iter().copied().min()?;
                let max_key = keys.iter().copied().max()?;
                (min_key, max_key, keys.len())
            }
        };
        selective_i64_range(min_key, max_key, len)
    }

    pub(crate) fn insert(&mut self, key: i64) {
        match self {
            Self::Dense {
                contains,
                words,
                len,
                max_dense_key,
            } => {
                let Some(index) = adaptive_dense_index(key, *max_dense_key) else {
                    let mut keys = adaptive_i64_set_dense_to_hash(contains);
                    keys.insert(key);
                    *self = Self::Hash(keys);
                    return;
                };
                if index >= contains.len() {
                    contains.resize(index + 1, false);
                    words.resize(index / 64 + 1, 0);
                }
                if !contains[index] {
                    contains[index] = true;
                    words[index / 64] |= 1_u64 << (index % 64);
                    *len += 1;
                }
            }
            Self::Hash(keys) => {
                keys.insert(key);
            }
        }
    }

    pub(crate) fn try_insert_dense_values(&mut self, values: &[i64]) -> bool {
        let Self::Dense {
            contains,
            words,
            len,
            max_dense_key,
        } = self
        else {
            return false;
        };
        let mut max_index = 0_usize;
        for &key in values {
            let Some(index) = adaptive_dense_index(key, *max_dense_key) else {
                return false;
            };
            max_index = max_index.max(index);
        }
        if max_index >= contains.len() {
            contains.resize(max_index + 1, false);
            words.resize(max_index / 64 + 1, 0);
        }
        for &key in values {
            let index = usize::try_from(key).expect("validated dense key");
            if !contains[index] {
                contains[index] = true;
                words[index / 64] |= 1_u64 << (index % 64);
                *len += 1;
            }
        }
        true
    }

    pub(crate) fn from_hash(values: HashSet<i64>) -> Self {
        if values.is_empty() {
            return Self::new_dense();
        }
        let mut max_key = 0_usize;
        let max_dense_key = default_max_dense_i64_set_key();
        for key in values.iter().copied() {
            let Some(index) = adaptive_dense_index(key, max_dense_key) else {
                return Self::Hash(values.into_iter().collect());
            };
            max_key = max_key.max(index);
        }
        let mut contains = vec![false; max_key + 1];
        let mut len = 0_usize;
        for key in values {
            let index = usize::try_from(key).expect("validated dense key");
            if !contains[index] {
                contains[index] = true;
                len += 1;
            }
        }
        let words = adaptive_i64_set_words_from_contains(&contains);
        Self::Dense {
            contains,
            words,
            len,
            max_dense_key,
        }
    }
}

fn default_max_dense_i64_set_key() -> usize {
    let max_bytes = std::env::var("DODAM_DENSE_I64_SET_BYTES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_MAX_DENSE_I64_SET_BYTES);
    max_bytes.saturating_mul(4).saturating_sub(1)
}

fn adaptive_i64_set_words_from_contains(contains: &[bool]) -> Vec<u64> {
    let mut words = vec![0_u64; contains.len().div_ceil(64)];
    for (index, present) in contains.iter().copied().enumerate() {
        if present {
            words[index / 64] |= 1_u64 << (index % 64);
        }
    }
    words
}

pub(crate) fn adaptive_i64_words_contains(words: &[u64], key: i64) -> bool {
    let Ok(index) = usize::try_from(key) else {
        return false;
    };
    words
        .get(index / 64)
        .is_some_and(|word| ((*word >> (index % 64)) & 1) != 0)
}

#[derive(Clone)]
pub(crate) struct DenseI64Probe {
    words: Vec<u64>,
}

impl DenseI64Probe {
    pub(crate) fn from_keys_with_max_key<I>(keys: I, max_key: usize) -> Option<Self>
    where
        I: IntoIterator<Item = i64>,
    {
        let mut words = vec![0_u64; (max_key + 1).div_ceil(64)];
        for key in keys {
            let Ok(index) = usize::try_from(key) else {
                return None;
            };
            if index > max_key {
                return None;
            }
            words[index / 64] |= 1_u64 << (index % 64);
        }
        Some(Self { words })
    }

    pub(crate) fn contains(&self, key: i64) -> bool {
        adaptive_i64_words_contains(&self.words, key)
    }
}

#[derive(Clone)]
pub(crate) struct DenseI64RankMap<T> {
    words: Vec<u64>,
    rank_prefixes: Vec<u32>,
    values: Vec<T>,
    min_key: i64,
    max_key: i64,
}

struct DenseI64RankChunkBitmap {
    words: Vec<u64>,
    base_word: usize,
    min_key: i64,
    max_key: i64,
    pair_count: usize,
}

impl<T> DenseI64RankMap<T>
where
    T: Copy + Default,
{
    pub(crate) fn from_pairs<I>(pairs: I, max_bytes: usize) -> Option<Self>
    where
        I: Iterator<Item = (i64, T)> + Clone,
    {
        let mut min_key = i64::MAX;
        let mut max_key = i64::MIN;
        let mut pair_count = 0usize;
        for (key, _) in pairs.clone() {
            usize::try_from(key).ok()?;
            min_key = min_key.min(key);
            max_key = max_key.max(key);
            pair_count = pair_count.checked_add(1)?;
        }
        if pair_count == 0 {
            return None;
        }
        let max_index = usize::try_from(max_key).ok()?;
        let word_count = max_index.checked_add(1)?.div_ceil(64);
        let bitmap_bytes = word_count.checked_mul(std::mem::size_of::<u64>())?;
        let rank_bytes = word_count.checked_mul(std::mem::size_of::<u32>())?;
        let value_bytes = pair_count.checked_mul(std::mem::size_of::<T>())?;
        let estimated_bytes = bitmap_bytes
            .checked_add(rank_bytes)?
            .checked_add(value_bytes)?;
        if estimated_bytes > max_bytes {
            return None;
        }

        let mut words = vec![0_u64; word_count];
        for (key, _) in pairs.clone() {
            let index = usize::try_from(key).ok()?;
            words[index / 64] |= 1_u64 << (index % 64);
        }

        let mut rank_prefixes = Vec::with_capacity(word_count);
        let mut value_count = 0usize;
        for word in words.iter().copied() {
            rank_prefixes.push(u32::try_from(value_count).ok()?);
            value_count = value_count.checked_add(word.count_ones() as usize)?;
        }
        u32::try_from(value_count).ok()?;
        let mut values = vec![T::default(); value_count];
        for (key, value) in pairs {
            let index = usize::try_from(key).ok()?;
            let word_index = index / 64;
            let bit_index = index % 64;
            let lower_bits = words[word_index] & ((1_u64 << bit_index).wrapping_sub(1));
            let rank = rank_prefixes[word_index] as usize + lower_bits.count_ones() as usize;
            values[rank] = value;
        }
        Some(Self {
            words,
            rank_prefixes,
            values,
            min_key,
            max_key,
        })
    }

    pub(crate) fn from_chunks_parallel<C>(chunks: &[C], max_bytes: usize) -> Option<Self>
    where
        C: AsRef<[(i64, T)]> + Sync,
        T: Send + Sync,
    {
        let chunk_bitmaps = chunks
            .par_iter()
            .map(|chunk| dense_i64_rank_chunk_bitmap(chunk.as_ref(), max_bytes))
            .collect::<Vec<_>>();
        let mut chunk_bitmaps = chunk_bitmaps
            .into_iter()
            .collect::<Option<Vec<DenseI64RankChunkBitmap>>>()?;
        let mut min_key = i64::MAX;
        let mut max_key = i64::MIN;
        let mut pair_count = 0usize;
        let mut local_word_count = 0usize;
        for chunk in &chunk_bitmaps {
            if chunk.pair_count == 0 {
                continue;
            }
            min_key = min_key.min(chunk.min_key);
            max_key = max_key.max(chunk.max_key);
            pair_count = pair_count.checked_add(chunk.pair_count)?;
            local_word_count = local_word_count.checked_add(chunk.words.len())?;
        }
        if pair_count == 0 {
            return None;
        }
        let max_index = usize::try_from(max_key).ok()?;
        let word_count = max_index.checked_add(1)?.div_ceil(64);
        let bitmap_bytes = word_count.checked_mul(std::mem::size_of::<u64>())?;
        let rank_bytes = word_count.checked_mul(std::mem::size_of::<u32>())?;
        let value_bytes = pair_count.checked_mul(std::mem::size_of::<T>())?;
        let estimated_bytes = bitmap_bytes
            .checked_add(rank_bytes)?
            .checked_add(value_bytes)?;
        let local_bitmap_bytes = local_word_count.checked_mul(std::mem::size_of::<u64>())?;
        if estimated_bytes > max_bytes || local_bitmap_bytes > max_bytes {
            return None;
        }

        let ordered_non_overlapping = chunk_bitmaps
            .iter()
            .filter(|chunk| chunk.pair_count > 0)
            .try_fold(None, |previous_max, chunk| {
                if previous_max.is_some_and(|max_key| max_key >= chunk.min_key) {
                    None
                } else {
                    Some(Some(chunk.max_key))
                }
            })
            .is_some();
        let mut words = vec![0_u64; word_count];
        for chunk in &chunk_bitmaps {
            for (offset, word) in chunk.words.iter().copied().enumerate() {
                words[chunk.base_word + offset] |= word;
            }
        }
        chunk_bitmaps.clear();
        chunk_bitmaps.shrink_to_fit();

        let mut rank_prefixes = Vec::with_capacity(word_count);
        let mut value_count = 0usize;
        for word in words.iter().copied() {
            rank_prefixes.push(u32::try_from(value_count).ok()?);
            value_count = value_count.checked_add(word.count_ones() as usize)?;
        }
        u32::try_from(value_count).ok()?;

        let values = if ordered_non_overlapping && value_count == pair_count {
            let chunk_values = chunks
                .par_iter()
                .map(|chunk| {
                    let pairs = chunk.as_ref();
                    if pairs.is_empty() {
                        return Some(Vec::new());
                    }
                    let min_key = pairs.iter().map(|(key, _)| *key).min()?;
                    let base_rank = dense_i64_rank_from_parts(&words, &rank_prefixes, min_key)?;
                    let mut values = vec![T::default(); pairs.len()];
                    for (key, value) in pairs.iter().copied() {
                        let rank = dense_i64_rank_from_parts(&words, &rank_prefixes, key)?;
                        *values.get_mut(rank.checked_sub(base_rank)?)? = value;
                    }
                    Some(values)
                })
                .collect::<Vec<_>>();
            let mut values = Vec::with_capacity(value_count);
            for mut chunk_values in chunk_values.into_iter().collect::<Option<Vec<_>>>()? {
                values.append(&mut chunk_values);
            }
            (values.len() == value_count).then_some(values)?
        } else {
            let mut values = vec![T::default(); value_count];
            for (key, value) in chunks
                .iter()
                .flat_map(|chunk| chunk.as_ref().iter().copied())
            {
                let rank = dense_i64_rank_from_parts(&words, &rank_prefixes, key)?;
                values[rank] = value;
            }
            values
        };
        Some(Self {
            words,
            rank_prefixes,
            values,
            min_key,
            max_key,
        })
    }

    pub(crate) fn get(&self, key: i64) -> Option<T> {
        let index = usize::try_from(key).ok()?;
        let word_index = index / 64;
        let bit_index = index % 64;
        let word = *self.words.get(word_index)?;
        let bit = 1_u64 << bit_index;
        if word & bit == 0 {
            return None;
        }
        let lower_bits = word & bit.wrapping_sub(1);
        let rank = *self.rank_prefixes.get(word_index)? as usize + lower_bits.count_ones() as usize;
        self.values.get(rank).copied()
    }

    pub(crate) fn contains_key(&self, key: i64) -> bool {
        self.get(key).is_some()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub(crate) fn max_key(&self) -> i64 {
        self.max_key
    }

    pub(crate) fn selective_key_range(&self) -> Option<(i64, i64)> {
        selective_i64_range(self.min_key, self.max_key, self.values.len())
    }
}

fn dense_i64_rank_chunk_bitmap<T: Copy>(
    pairs: &[(i64, T)],
    max_bytes: usize,
) -> Option<DenseI64RankChunkBitmap> {
    if pairs.is_empty() {
        return Some(DenseI64RankChunkBitmap {
            words: Vec::new(),
            base_word: 0,
            min_key: 0,
            max_key: -1,
            pair_count: 0,
        });
    }
    let mut min_key = i64::MAX;
    let mut max_key = i64::MIN;
    for (key, _) in pairs.iter().copied() {
        usize::try_from(key).ok()?;
        min_key = min_key.min(key);
        max_key = max_key.max(key);
    }
    let base_word = usize::try_from(min_key).ok()? / 64;
    let max_word = usize::try_from(max_key).ok()? / 64;
    let word_count = max_word.checked_sub(base_word)?.checked_add(1)?;
    if word_count.checked_mul(std::mem::size_of::<u64>())? > max_bytes {
        return None;
    }
    let mut words = vec![0_u64; word_count];
    for (key, _) in pairs.iter().copied() {
        let index = usize::try_from(key).ok()?;
        words[index / 64 - base_word] |= 1_u64 << (index % 64);
    }
    Some(DenseI64RankChunkBitmap {
        words,
        base_word,
        min_key,
        max_key,
        pair_count: pairs.len(),
    })
}

fn dense_i64_rank_from_parts(words: &[u64], rank_prefixes: &[u32], key: i64) -> Option<usize> {
    let index = usize::try_from(key).ok()?;
    let word_index = index / 64;
    let bit_index = index % 64;
    let word = *words.get(word_index)?;
    let bit = 1_u64 << bit_index;
    if word & bit == 0 {
        return None;
    }
    let lower_bits = word & bit.wrapping_sub(1);
    Some(*rank_prefixes.get(word_index)? as usize + lower_bits.count_ones() as usize)
}

pub(crate) struct PackedU32PairDistinct {
    pairs: Vec<u64>,
}

impl PackedU32PairDistinct {
    pub(crate) fn new() -> Self {
        Self { pairs: Vec::new() }
    }

    pub(crate) fn push(&mut self, first: usize, second: i64) -> bool {
        let Some(key) = pack_u32_pair(first, second) else {
            return false;
        };
        self.pairs.push(key);
        true
    }

    pub(crate) fn append(&mut self, other: &mut Self) {
        self.pairs.append(&mut other.pairs);
    }

    pub(crate) fn counts_by_first(mut self, first_count: usize) -> Vec<u64> {
        self.pairs.sort_unstable();
        self.pairs.dedup();
        let mut counts = vec![0_u64; first_count];
        for key in self.pairs {
            if let Some(count) = counts.get_mut(unpack_u32_pair_first(key)) {
                *count += 1;
            }
        }
        counts
    }
}

fn pack_u32_pair(first: usize, second: i64) -> Option<u64> {
    let first = u32::try_from(first).ok()?;
    let second = u32::try_from(second).ok()?;
    Some((u64::from(first) << 32) | u64::from(second))
}

fn unpack_u32_pair_first(key: u64) -> usize {
    (key >> 32) as usize
}

#[derive(Clone, Default)]
pub(crate) struct DenseI64BoolLookup {
    present: Vec<bool>,
    values: Vec<bool>,
    len: usize,
}

impl DenseI64BoolLookup {
    pub(crate) fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub(crate) fn insert(&mut self, key: i64, value: bool) {
        let Some(index) = adaptive_dense_index(key, DEFAULT_MAX_DENSE_I64_KEY) else {
            return;
        };
        if index >= self.present.len() {
            self.present.resize(index + 1, false);
            self.values.resize(index + 1, false);
        }
        if !self.present[index] {
            self.present[index] = true;
            self.len += 1;
        }
        self.values[index] = value;
    }

    pub(crate) fn get(&self, key: i64) -> Option<bool> {
        let index = usize::try_from(key).ok()?;
        if index < self.present.len() && self.present[index] {
            Some(self.values[index])
        } else {
            None
        }
    }
}

fn adaptive_i64_set_dense_to_hash(contains: &[bool]) -> FastHashSet<i64> {
    contains
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(key, contains)| contains.then_some(key as i64))
        .collect()
}

#[derive(Clone)]
pub(crate) enum AdaptiveI64Map<V> {
    Dense {
        values: Vec<V>,
        present: Vec<bool>,
        words: Vec<u64>,
        len: usize,
    },
    Hash(FastHashMap<i64, V>),
}

impl<V> AdaptiveI64Map<V>
where
    V: Copy + Default,
{
    pub(crate) fn new_dense() -> Self {
        Self::Dense {
            values: Vec::new(),
            present: Vec::new(),
            words: Vec::new(),
            len: 0,
        }
    }

    pub(crate) fn from_hash<S>(values: HashMap<i64, V, S>) -> Self
    where
        S: BuildHasher,
    {
        if values.is_empty() {
            return Self::new_dense();
        }
        let mut max_key = 0_usize;
        for key in values.keys().copied() {
            let Some(index) = adaptive_dense_index(key, DEFAULT_MAX_DENSE_I64_KEY) else {
                return Self::Hash(values.into_iter().collect());
            };
            max_key = max_key.max(index);
        }
        let mut dense_values = vec![V::default(); max_key + 1];
        let mut present = vec![false; max_key + 1];
        let mut words = vec![0_u64; (max_key + 1).div_ceil(64)];
        let mut len = 0_usize;
        for (key, value) in values {
            let index = usize::try_from(key).expect("validated dense key");
            if !present[index] {
                present[index] = true;
                words[index / 64] |= 1_u64 << (index % 64);
                len += 1;
            }
            dense_values[index] = value;
        }
        Self::Dense {
            values: dense_values,
            present,
            words,
            len,
        }
    }

    pub(crate) fn get(&self, key: i64) -> Option<V> {
        match self {
            Self::Dense { values, words, .. } => {
                let index = usize::try_from(key).ok()?;
                words
                    .get(index / 64)
                    .copied()
                    .filter(|word| ((*word >> (index % 64)) & 1) != 0)
                    .and_then(|_| values.get(index).copied())
            }
            Self::Hash(values) => values.get(&key).copied(),
        }
    }

    pub(crate) fn dense_slices(&self) -> Option<(&[V], &[bool])> {
        match self {
            Self::Dense {
                values, present, ..
            } => Some((values, present)),
            Self::Hash(_) => None,
        }
    }

    pub(crate) fn dense_word_slices(&self) -> Option<(&[V], &[u64])> {
        match self {
            Self::Dense { values, words, .. } => Some((values, words)),
            Self::Hash(_) => None,
        }
    }

    pub(crate) fn get_cached(&self, dense_slices: Option<(&[V], &[bool])>, key: i64) -> Option<V> {
        if let Some((values, present)) = dense_slices {
            let index = usize::try_from(key).ok()?;
            return present
                .get(index)
                .copied()
                .filter(|present| *present)
                .map(|_| values[index]);
        }
        self.get(key)
    }

    pub(crate) fn get_cached_words(
        &self,
        dense_slices: Option<(&[V], &[u64])>,
        key: i64,
    ) -> Option<V> {
        if let Some((values, words)) = dense_slices {
            let index = usize::try_from(key).ok()?;
            return words
                .get(index / 64)
                .copied()
                .filter(|word| ((*word >> (index % 64)) & 1) != 0)
                .and_then(|_| values.get(index).copied());
        }
        self.get(key)
    }

    pub(crate) fn is_empty(&self) -> bool {
        match self {
            Self::Dense { len, .. } => *len == 0,
            Self::Hash(values) => values.is_empty(),
        }
    }

    pub(crate) fn selective_key_range(&self) -> Option<(i64, i64)> {
        let (min_key, max_key, len) = match self {
            Self::Dense { present, len, .. } => {
                let min_key = present.iter().position(|present| *present)? as i64;
                let max_key = present.iter().rposition(|present| *present)? as i64;
                (min_key, max_key, *len)
            }
            Self::Hash(values) => {
                let min_key = values.keys().copied().min()?;
                let max_key = values.keys().copied().max()?;
                (min_key, max_key, values.len())
            }
        };
        selective_i64_range(min_key, max_key, len)
    }

    pub(crate) fn insert(&mut self, key: i64, value: V) {
        match self {
            Self::Dense {
                values,
                present,
                words,
                len,
            } => {
                let Some(index) = adaptive_dense_index(key, DEFAULT_MAX_DENSE_I64_KEY) else {
                    let mut hash = adaptive_i64_map_dense_to_hash(values, present);
                    hash.insert(key, value);
                    *self = Self::Hash(hash);
                    return;
                };
                if index >= values.len() {
                    values.resize(index + 1, V::default());
                    present.resize(index + 1, false);
                    words.resize(index / 64 + 1, 0);
                }
                if !present[index] {
                    present[index] = true;
                    words[index / 64] |= 1_u64 << (index % 64);
                    *len += 1;
                }
                values[index] = value;
            }
            Self::Hash(values) => {
                values.insert(key, value);
            }
        }
    }

    pub(crate) fn update<Init, Update>(&mut self, key: i64, init: Init, update: Update)
    where
        Init: FnOnce() -> V,
        Update: FnOnce(&mut V),
    {
        match self {
            Self::Dense {
                values,
                present,
                words,
                len,
            } => {
                let Some(index) = adaptive_dense_index(key, DEFAULT_MAX_DENSE_I64_KEY) else {
                    let mut hash = adaptive_i64_map_dense_to_hash(values, present);
                    let entry = hash.entry(key).or_insert_with(init);
                    update(entry);
                    *self = Self::Hash(hash);
                    return;
                };
                if index >= values.len() {
                    values.resize(index + 1, V::default());
                    present.resize(index + 1, false);
                    words.resize(index / 64 + 1, 0);
                }
                if !present[index] {
                    values[index] = init();
                    present[index] = true;
                    words[index / 64] |= 1_u64 << (index % 64);
                    *len += 1;
                }
                update(&mut values[index]);
            }
            Self::Hash(values) => {
                let entry = values.entry(key).or_insert_with(init);
                update(entry);
            }
        }
    }

    pub(crate) fn into_filtered_hash<P>(self, predicate: P) -> HashMap<i64, V>
    where
        P: Fn(V) -> bool,
    {
        match self {
            Self::Dense {
                values, present, ..
            } => values
                .into_iter()
                .zip(present)
                .enumerate()
                .filter_map(|(key, (value, present))| {
                    (present && predicate(value)).then_some((key as i64, value))
                })
                .collect(),
            Self::Hash(values) => values
                .into_iter()
                .filter(|(_, value)| predicate(*value))
                .collect(),
        }
    }
}

pub(crate) struct DenseI64F64Sum {
    dense: Vec<f64>,
    fallback: Option<AdaptiveI64Map<f64>>,
    threshold: Option<f64>,
    threshold_candidates: Vec<i64>,
    all_non_negative: bool,
}

impl DenseI64F64Sum {
    pub(crate) fn new() -> Self {
        Self {
            dense: Vec::new(),
            fallback: None,
            threshold: None,
            threshold_candidates: Vec::new(),
            all_non_negative: true,
        }
    }

    pub(crate) fn new_tracking_threshold(threshold: f64) -> Self {
        Self {
            threshold: Some(threshold),
            ..Self::new()
        }
    }

    pub(crate) fn add_dense_index(&mut self, index: usize, value: f64) {
        debug_assert!(self.fallback.is_none());
        if value < 0.0 {
            self.all_non_negative = false;
        }
        let previous = self.dense[index];
        self.dense[index] += value;
        if let Some(threshold) = self.threshold
            && previous <= threshold
            && self.dense[index] > threshold
        {
            self.threshold_candidates.push(index as i64);
        }
    }

    pub(crate) fn reserve_dense_to(&mut self, max_key: usize) {
        if self.fallback.is_none() && max_key >= self.dense.len() {
            self.dense.resize(max_key + 1, 0.0);
        }
    }

    pub(crate) fn try_reserve_dense_to(&mut self, max_key: usize, max_bytes: usize) -> bool {
        if self.fallback.is_some() {
            return false;
        }
        let Some(required_entries) = max_key.checked_add(1) else {
            return false;
        };
        let Some(required_bytes) = required_entries.checked_mul(std::mem::size_of::<f64>()) else {
            return false;
        };
        if required_bytes > max_bytes {
            return false;
        }
        if required_entries > self.dense.len() {
            let additional = required_entries - self.dense.len();
            if self.dense.try_reserve_exact(additional).is_err() {
                return false;
            }
            self.dense.resize(required_entries, 0.0);
        }
        true
    }

    pub(crate) fn try_add_dense_key(&mut self, key: i64, value: f64) -> bool {
        let Ok(index) = usize::try_from(key) else {
            return false;
        };
        if index >= self.dense.len() {
            return false;
        }
        self.add_dense_index(index, value);
        true
    }

    pub(crate) fn has_fallback(&self) -> bool {
        self.fallback.is_some()
    }

    pub(crate) fn fallback_mut(&mut self) -> Option<&mut AdaptiveI64Map<f64>> {
        self.fallback.as_mut()
    }

    pub(crate) fn convert_to_fallback(&mut self) {
        if self.fallback.is_some() {
            return;
        }
        let mut fallback = AdaptiveI64Map::<f64>::new_dense();
        for (key, value) in self.dense.iter().copied().enumerate() {
            if value != 0.0 {
                fallback.insert(key as i64, value);
            }
        }
        self.dense.clear();
        self.fallback = Some(fallback);
        self.threshold_candidates.clear();
    }

    pub(crate) fn into_filtered_hash<P>(self, predicate: P) -> HashMap<i64, f64>
    where
        P: Fn(f64) -> bool,
    {
        if let Some(fallback) = self.fallback {
            return fallback.into_filtered_hash(predicate);
        }
        if self.threshold.is_some() && self.all_non_negative {
            return self
                .threshold_candidates
                .iter()
                .copied()
                .filter_map(|key| {
                    let value = self.dense.get(usize::try_from(key).ok()?).copied()?;
                    predicate(value).then_some((key, value))
                })
                .collect();
        }
        self.dense
            .into_iter()
            .enumerate()
            .filter_map(|(key, value)| predicate(value).then_some((key as i64, value)))
            .collect()
    }
}

#[derive(Clone)]
pub(crate) struct DenseI64Map<V> {
    dense: Vec<V>,
    base_key: i64,
    missing: V,
    fallback: Option<AdaptiveI64Map<V>>,
}

impl<V> DenseI64Map<V>
where
    V: Copy + Default + PartialEq,
{
    pub(crate) fn new(missing: V) -> Self {
        Self {
            dense: Vec::new(),
            base_key: 0,
            missing,
            fallback: None,
        }
    }

    pub(crate) fn from_pairs_with_dense_range_policy(
        pairs: Vec<(i64, V)>,
        missing: V,
        max_entries: usize,
        max_amplification: f64,
    ) -> Self {
        let mut map = Self::new(missing);
        if pairs.is_empty() {
            return map;
        }
        let mut min_key = i64::MAX;
        let mut max_key = i64::MIN;
        let mut fallback_required = false;
        for (key, _) in pairs.iter().copied() {
            if key < 0 {
                fallback_required = true;
                break;
            }
            min_key = min_key.min(key);
            max_key = max_key.max(key);
        }
        if !fallback_required
            && dense_i64_range_within_amplification(
                min_key,
                max_key,
                pairs.len(),
                max_amplification,
            )
            && map.reserve_dense_range(min_key, max_key, max_entries)
        {
            for (key, value) in pairs {
                map.insert_dense_key(key, value);
            }
            return map;
        }
        let mut fallback = fast_hash_map_with_capacity(pairs.len());
        for (key, value) in pairs {
            fallback.insert(key, value);
        }
        map.fallback = Some(AdaptiveI64Map::Hash(fallback));
        map
    }

    pub(crate) fn get(&self, key: i64) -> Option<V> {
        if let Some(fallback) = self.fallback.as_ref() {
            return fallback.get(key);
        }
        let index = usize::try_from(key.checked_sub(self.base_key)?).ok()?;
        self.dense
            .get(index)
            .copied()
            .filter(|value| *value != self.missing)
    }

    pub(crate) fn dense_slice(&self) -> Option<(&[V], i64, V)> {
        self.fallback
            .is_none()
            .then_some((self.dense.as_slice(), self.base_key, self.missing))
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub(crate) fn len(&self) -> usize {
        if let Some(fallback) = self.fallback.as_ref() {
            return match fallback {
                AdaptiveI64Map::Dense { len, .. } => *len,
                AdaptiveI64Map::Hash(values) => values.len(),
            };
        }
        self.dense
            .iter()
            .copied()
            .filter(|value| *value != self.missing)
            .count()
    }

    pub(crate) fn selective_key_range(&self) -> Option<(i64, i64)> {
        if let Some(fallback) = self.fallback.as_ref() {
            let (min_key, max_key, len) = match fallback {
                AdaptiveI64Map::Dense { present, len, .. } => {
                    let min_key = present.iter().position(|present| *present)? as i64;
                    let max_key = present.iter().rposition(|present| *present)? as i64;
                    (min_key, max_key, *len)
                }
                AdaptiveI64Map::Hash(values) => {
                    let min_key = values.keys().copied().min()?;
                    let max_key = values.keys().copied().max()?;
                    (min_key, max_key, values.len())
                }
            };
            return selective_i64_range(min_key, max_key, len);
        }
        let min_offset = self.dense.iter().position(|value| *value != self.missing)?;
        let max_offset = self
            .dense
            .iter()
            .rposition(|value| *value != self.missing)?;
        selective_i64_range(
            self.base_key + min_offset as i64,
            self.base_key + max_offset as i64,
            self.len(),
        )
    }

    pub(crate) fn reserve_dense_range(
        &mut self,
        min_key: i64,
        max_key: i64,
        max_entries: usize,
    ) -> bool {
        debug_assert!(self.fallback.is_none());
        if min_key < 0 || max_key < min_key {
            return false;
        }
        let (new_min, new_max) = if self.dense.is_empty() {
            (min_key, max_key)
        } else {
            let current_max = self
                .base_key
                .checked_add(self.dense.len() as i64)
                .and_then(|value| value.checked_sub(1));
            let Some(current_max) = current_max else {
                return false;
            };
            (self.base_key.min(min_key), current_max.max(max_key))
        };
        let Some(width) = new_max
            .checked_sub(new_min)
            .and_then(|width| width.checked_add(1))
            .and_then(|width| usize::try_from(width).ok())
        else {
            return false;
        };
        if width > max_entries {
            return false;
        }
        if self.dense.is_empty() {
            self.base_key = new_min;
            self.dense.resize(width, self.missing);
            return true;
        }
        if new_min == self.base_key {
            if width > self.dense.len() {
                self.dense.resize(width, self.missing);
            }
            return true;
        }
        let offset = usize::try_from(self.base_key - new_min).expect("validated dense offset");
        let mut dense = vec![self.missing; width];
        dense[offset..offset + self.dense.len()].copy_from_slice(&self.dense);
        self.base_key = new_min;
        self.dense = dense;
        true
    }

    pub(crate) fn insert_dense_key(&mut self, key: i64, value: V) {
        debug_assert!(self.fallback.is_none());
        if let Some(delta) = key.checked_sub(self.base_key)
            && let Ok(index) = usize::try_from(delta)
            && index < self.dense.len()
        {
            self.dense[index] = value;
        }
    }

    pub(crate) fn fallback_mut(&mut self) -> Option<&mut AdaptiveI64Map<V>> {
        self.fallback.as_mut()
    }

    pub(crate) fn convert_to_fallback(&mut self) {
        if self.fallback.is_some() {
            return;
        }
        let mut fallback = AdaptiveI64Map::<V>::new_dense();
        for (key, value) in self.dense.iter().copied().enumerate() {
            if value != self.missing {
                fallback.insert(self.base_key + key as i64, value);
            }
        }
        self.dense.clear();
        self.fallback = Some(fallback);
    }
}

pub(crate) type DenseI64I32Map = DenseI64Map<i32>;
pub(crate) type DenseI64U8Map = DenseI64Map<u8>;

pub(crate) fn dense_i64_range_within_amplification(
    min_key: i64,
    max_key: i64,
    row_count: usize,
    max_amplification: f64,
) -> bool {
    if row_count == 0 {
        return true;
    }
    if !max_amplification.is_finite() || max_amplification < 1.0 {
        return false;
    }
    let Some(width) = max_key
        .checked_sub(min_key)
        .and_then(|width| width.checked_add(1))
    else {
        return false;
    };
    (width as f64) <= (row_count as f64) * max_amplification
}

fn adaptive_i64_map_dense_to_hash<V>(values: &[V], present: &[bool]) -> FastHashMap<i64, V>
where
    V: Copy,
{
    values
        .iter()
        .copied()
        .zip(present.iter().copied())
        .enumerate()
        .filter_map(|(key, (value, present))| present.then_some((key as i64, value)))
        .collect()
}

fn selective_i64_range(min_key: i64, max_key: i64, len: usize) -> Option<(i64, i64)> {
    if len == 0 || min_key < 0 || max_key < min_key {
        return None;
    }
    let width = usize::try_from(max_key.checked_sub(min_key)?.checked_add(1)?).ok()?;
    (width <= len.saturating_mul(8).max(1024)).then_some((min_key, max_key))
}

pub(crate) fn adaptive_dense_index(key: i64, max_dense_key: usize) -> Option<usize> {
    let index = usize::try_from(key).ok()?;
    (index <= max_dense_key).then_some(index)
}

#[cfg(test)]
mod tests {
    use super::{
        AdaptiveI64Set, DEFAULT_MAX_DENSE_I64_KEY, DenseI64F64Sum, DenseI64RankMap, DenseI64U8Map,
    };

    #[test]
    fn dense_i64_f64_sum_respects_byte_budget() {
        let mut sums = DenseI64F64Sum::new();
        assert!(!sums.try_reserve_dense_to(8, 8 * std::mem::size_of::<f64>()));
        assert!(sums.try_reserve_dense_to(7, 8 * std::mem::size_of::<f64>()));
        assert!(sums.try_add_dense_key(7, 2.5));
        assert!(!sums.try_add_dense_key(8, 1.0));

        let values = sums.into_filtered_hash(|value| value != 0.0);
        assert_eq!(values.get(&7), Some(&2.5));
    }

    #[test]
    fn adaptive_i64_set_uses_its_bit_budget_beyond_dense_map_limit() {
        let key = DEFAULT_MAX_DENSE_I64_KEY as i64 + 1;
        let mut set = AdaptiveI64Set::new_dense();
        set.insert(key);

        assert!(set.dense_word_slice().is_some());
        assert!(set.contains(key));
    }

    #[test]
    fn adaptive_i64_set_word_probe_matches_dense_membership() {
        let mut set = AdaptiveI64Set::new_dense();
        for key in [0_i64, 1, 63, 64, 65, 127, 130] {
            set.insert(key);
        }

        let dense_contains = set.dense_contains_slice();
        for key in [-1_i64, 0, 2, 63, 64, 66, 127, 128, 130, 131] {
            assert_eq!(set.contains(key), set.contains_cached(dense_contains, key));
        }
    }

    #[test]
    fn dense_i64_u8_map_uses_value_width_for_dense_budget() {
        let pairs = vec![(100_i64, 1_u8), (102, 7)];
        let dense = DenseI64U8Map::from_pairs_with_dense_range_policy(pairs.clone(), 0, 3, 2.0);
        let (values, base_key, missing) = dense.dense_slice().expect("dense u8 map");
        assert_eq!(base_key, 100);
        assert_eq!(missing, 0);
        assert_eq!(values, &[1, 0, 7]);
        assert_eq!(dense.get(100), Some(1));
        assert_eq!(dense.get(101), None);
        assert_eq!(dense.get(102), Some(7));

        let fallback = DenseI64U8Map::from_pairs_with_dense_range_policy(pairs, 0, 2, 2.0);
        assert!(fallback.dense_slice().is_none());
        assert_eq!(fallback.get(100), Some(1));
        assert_eq!(fallback.get(102), Some(7));
    }

    #[test]
    fn dense_i64_rank_map_handles_sparse_keys_and_duplicate_values() {
        let pairs = vec![(1_i64, 10_i32), (64, 20), (130, 30), (64, 21)];
        let map = DenseI64RankMap::from_pairs(pairs.iter().copied(), 4096).unwrap();

        assert_eq!(map.get(1), Some(10));
        assert_eq!(map.get(64), Some(21));
        assert_eq!(map.get(130), Some(30));
        assert_eq!(map.get(0), None);
        assert_eq!(map.get(129), None);
        assert_eq!(map.get(-1), None);
        assert!(map.contains_key(64));
        assert!(!map.is_empty());
        assert_eq!(map.max_key(), 130);
    }

    #[test]
    fn dense_i64_rank_map_builds_ordered_chunks_in_parallel() {
        let chunks = vec![
            vec![(64_i64, 20_i32), (1, 10)],
            Vec::new(),
            vec![(130, 30), (129, 29)],
        ];
        let map = DenseI64RankMap::from_chunks_parallel(&chunks, 4096).unwrap();

        assert_eq!(map.get(1), Some(10));
        assert_eq!(map.get(64), Some(20));
        assert_eq!(map.get(129), Some(29));
        assert_eq!(map.get(130), Some(30));
        assert_eq!(map.get(128), None);
    }

    #[test]
    fn dense_i64_rank_map_parallel_builder_preserves_duplicate_order() {
        let chunks = vec![vec![(1_i64, 10_i32), (64, 20)], vec![(64, 21), (2, 30)]];
        let map = DenseI64RankMap::from_chunks_parallel(&chunks, 4096).unwrap();

        assert_eq!(map.get(1), Some(10));
        assert_eq!(map.get(2), Some(30));
        assert_eq!(map.get(64), Some(21));
    }

    #[test]
    fn dense_i64_rank_map_respects_memory_budget() {
        let pairs = vec![(0_i64, 1_i32), (1024, 2)];
        assert!(DenseI64RankMap::from_pairs(pairs.iter().copied(), 8).is_none());
    }
}
