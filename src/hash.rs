use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};

pub(crate) type FastHashMap<K, V> = HashMap<K, V, BuildHasherDefault<FastHasher>>;
pub(crate) type FastHashSet<K> = HashSet<K, BuildHasherDefault<FastHasher>>;

#[derive(Default)]
pub(crate) struct FastHasher {
    hash: u64,
}

impl Hasher for FastHasher {
    fn finish(&self) -> u64 {
        self.hash
    }

    fn write(&mut self, bytes: &[u8]) {
        let mut hash = self.hash ^ 0xcbf2_9ce4_8422_2325;
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x1000_0000_01b3);
        }
        self.hash = hash;
    }

    fn write_i32(&mut self, value: i32) {
        self.write_u64(value as u32 as u64);
    }

    fn write_u32(&mut self, value: u32) {
        self.write_u64(u64::from(value));
    }

    fn write_i64(&mut self, value: i64) {
        self.write_u64(value as u64);
    }

    fn write_u64(&mut self, value: u64) {
        let mut hash = self.hash ^ value;
        hash ^= hash >> 33;
        hash = hash.wrapping_mul(0xff51_afd7_ed55_8ccd);
        hash ^= hash >> 33;
        hash = hash.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
        hash ^= hash >> 33;
        self.hash = hash;
    }

    fn write_usize(&mut self, value: usize) {
        self.write_u64(value as u64);
    }
}

pub(crate) fn fast_hash_map<K, V>() -> FastHashMap<K, V> {
    HashMap::with_hasher(BuildHasherDefault::<FastHasher>::default())
}

pub(crate) fn fast_hash_map_with_capacity<K, V>(capacity: usize) -> FastHashMap<K, V> {
    HashMap::with_capacity_and_hasher(capacity, BuildHasherDefault::<FastHasher>::default())
}
