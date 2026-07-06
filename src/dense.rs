use std::collections::{HashMap, HashSet};
use std::hash::BuildHasher;

use crate::hash::{FastHashMap, FastHashSet};

const DEFAULT_MAX_DENSE_I64_KEY: usize = 20_000_000;

#[derive(Clone)]
pub(crate) enum AdaptiveI64Set {
    Dense { contains: Vec<bool>, len: usize },
    Hash(FastHashSet<i64>),
}

impl AdaptiveI64Set {
    pub(crate) fn new_dense() -> Self {
        Self::Dense {
            contains: Vec::new(),
            len: 0,
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
            Self::Dense { contains, .. } => usize::try_from(key)
                .ok()
                .and_then(|index| contains.get(index))
                .copied()
                .unwrap_or(false),
            Self::Hash(keys) => keys.contains(&key),
        }
    }

    pub(crate) fn dense_contains_slice(&self) -> Option<&[bool]> {
        match self {
            Self::Dense { contains, .. } => Some(contains),
            Self::Hash(_) => None,
        }
    }

    pub(crate) fn to_hash_set(&self) -> HashSet<i64> {
        match self {
            Self::Dense { contains, len } => {
                let mut keys = HashSet::with_capacity(*len);
                for (key, present) in contains.iter().copied().enumerate() {
                    if present {
                        keys.insert(key as i64);
                    }
                }
                keys
            }
            Self::Hash(keys) => keys.iter().copied().collect(),
        }
    }

    pub(crate) fn selective_key_range(&self) -> Option<(i64, i64)> {
        let (min_key, max_key, len) = match self {
            Self::Dense { contains, len } => {
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
            Self::Dense { contains, len } => {
                let Some(index) = adaptive_dense_index(key, DEFAULT_MAX_DENSE_I64_KEY) else {
                    let mut keys = adaptive_i64_set_dense_to_hash(contains);
                    keys.insert(key);
                    *self = Self::Hash(keys);
                    return;
                };
                if index >= contains.len() {
                    contains.resize(index + 1, false);
                }
                if !contains[index] {
                    contains[index] = true;
                    *len += 1;
                }
            }
            Self::Hash(keys) => {
                keys.insert(key);
            }
        }
    }

    pub(crate) fn try_insert_dense_values(&mut self, values: &[i64]) -> bool {
        let Self::Dense { contains, len } = self else {
            return false;
        };
        let mut max_index = 0_usize;
        for &key in values {
            let Some(index) = adaptive_dense_index(key, DEFAULT_MAX_DENSE_I64_KEY) else {
                return false;
            };
            max_index = max_index.max(index);
        }
        if max_index >= contains.len() {
            contains.resize(max_index + 1, false);
        }
        for &key in values {
            let index = usize::try_from(key).expect("validated dense key");
            if !contains[index] {
                contains[index] = true;
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
        for key in values.iter().copied() {
            let Some(index) = adaptive_dense_index(key, DEFAULT_MAX_DENSE_I64_KEY) else {
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
        Self::Dense { contains, len }
    }
}

#[derive(Clone)]
pub(crate) struct SortedI64Lookup<V> {
    entries: Vec<(i64, V)>,
}

impl<V: Copy> SortedI64Lookup<V> {
    pub(crate) fn from_hash_map<S>(values: &HashMap<i64, V, S>) -> Self
    where
        S: BuildHasher,
    {
        let mut entries = values
            .iter()
            .map(|(&key, &value)| (key, value))
            .collect::<Vec<_>>();
        entries.sort_unstable_by_key(|(key, _)| *key);
        Self { entries }
    }

    pub(crate) fn get(&self, key: i64) -> Option<V> {
        self.entries
            .binary_search_by_key(&key, |(entry_key, _)| *entry_key)
            .ok()
            .map(|index| self.entries[index].1)
    }
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
        let mut len = 0_usize;
        for (key, value) in values {
            let index = usize::try_from(key).expect("validated dense key");
            if !present[index] {
                present[index] = true;
                len += 1;
            }
            dense_values[index] = value;
        }
        Self::Dense {
            values: dense_values,
            present,
            len,
        }
    }

    pub(crate) fn get(&self, key: i64) -> Option<V> {
        match self {
            Self::Dense {
                values, present, ..
            } => {
                let index = usize::try_from(key).ok()?;
                present
                    .get(index)
                    .copied()
                    .filter(|present| *present)
                    .map(|_| values[index])
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

    pub(crate) fn is_empty(&self) -> bool {
        match self {
            Self::Dense { len, .. } => *len == 0,
            Self::Hash(values) => values.is_empty(),
        }
    }

    pub(crate) fn insert(&mut self, key: i64, value: V) {
        match self {
            Self::Dense {
                values,
                present,
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
                }
                if !present[index] {
                    present[index] = true;
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
                }
                if !present[index] {
                    values[index] = init();
                    present[index] = true;
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

pub(crate) struct DenseI64I32Map {
    dense: Vec<i32>,
    base_key: i64,
    missing: i32,
    fallback: Option<AdaptiveI64Map<i32>>,
}

impl DenseI64I32Map {
    pub(crate) fn new(missing: i32) -> Self {
        Self {
            dense: Vec::new(),
            base_key: 0,
            missing,
            fallback: None,
        }
    }

    pub(crate) fn get(&self, key: i64) -> Option<i32> {
        if let Some(fallback) = self.fallback.as_ref() {
            return fallback.get(key);
        }
        let index = usize::try_from(key.checked_sub(self.base_key)?).ok()?;
        self.dense
            .get(index)
            .copied()
            .filter(|value| *value != self.missing)
    }

    pub(crate) fn dense_slice(&self) -> Option<(&[i32], i64, i32)> {
        self.fallback
            .is_none()
            .then_some((self.dense.as_slice(), self.base_key, self.missing))
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

    pub(crate) fn insert_dense_key(&mut self, key: i64, value: i32) {
        debug_assert!(self.fallback.is_none());
        if let Some(delta) = key.checked_sub(self.base_key)
            && let Ok(index) = usize::try_from(delta)
            && index < self.dense.len()
        {
            self.dense[index] = value;
        }
    }

    pub(crate) fn fallback_mut(&mut self) -> Option<&mut AdaptiveI64Map<i32>> {
        self.fallback.as_mut()
    }

    pub(crate) fn convert_to_fallback(&mut self) {
        if self.fallback.is_some() {
            return;
        }
        let mut fallback = AdaptiveI64Map::<i32>::new_dense();
        for (key, value) in self.dense.iter().copied().enumerate() {
            if value != self.missing {
                fallback.insert(self.base_key + key as i64, value);
            }
        }
        self.dense.clear();
        self.fallback = Some(fallback);
    }
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
