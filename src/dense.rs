use std::collections::{HashMap, HashSet};

const DEFAULT_MAX_DENSE_I64_KEY: usize = 20_000_000;

#[derive(Clone)]
pub(crate) enum AdaptiveI64Set {
    Dense { contains: Vec<bool>, len: usize },
    Hash(HashSet<i64>),
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

    pub(crate) fn from_hash(values: HashSet<i64>) -> Self {
        if values.is_empty() {
            return Self::new_dense();
        }
        let mut max_key = 0_usize;
        for key in values.iter().copied() {
            let Some(index) = adaptive_dense_index(key, DEFAULT_MAX_DENSE_I64_KEY) else {
                return Self::Hash(values);
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

fn adaptive_i64_set_dense_to_hash(contains: &[bool]) -> HashSet<i64> {
    contains
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(key, contains)| contains.then_some(key as i64))
        .collect()
}

pub(crate) enum AdaptiveI64Map<V> {
    Dense {
        values: Vec<V>,
        present: Vec<bool>,
        len: usize,
    },
    Hash(HashMap<i64, V>),
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

    pub(crate) fn from_hash(values: HashMap<i64, V>) -> Self {
        if values.is_empty() {
            return Self::new_dense();
        }
        let mut max_key = 0_usize;
        for key in values.keys().copied() {
            let Some(index) = adaptive_dense_index(key, DEFAULT_MAX_DENSE_I64_KEY) else {
                return Self::Hash(values);
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

fn adaptive_i64_map_dense_to_hash<V>(values: &[V], present: &[bool]) -> HashMap<i64, V>
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

pub(crate) fn adaptive_dense_index(key: i64, max_dense_key: usize) -> Option<usize> {
    let index = usize::try_from(key).ok()?;
    (index <= max_dense_key).then_some(index)
}
