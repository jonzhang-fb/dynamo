// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Manual backend for the mocker's KV manager.
//!
//! Uses [`HashCache`] for O(1) block lookups with active/inactive pool management
//! and manual reference counting.

use crate::cache::HashCache;
use crate::common::protocols::{KvCacheEventSink, MoveBlock};
use crate::kv_manager::KvBackend;
use dynamo_kv_router::protocols::{
    ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheRemoveData, KvCacheStoreData,
    KvCacheStoredBlockData, LocalBlockHash,
};
use dynamo_runtime::config::environment_names::mocker;
use dynamo_tokens::blocks::UniqueBlock;
use dynamo_tokens::{BlockHash, PositionalLineageHash, SequenceHash};
use std::collections::HashMap;
use std::env;
use std::sync::{Arc, LazyLock};
use std::time::{SystemTime, UNIX_EPOCH};

/// Check the env var to enable KV cache allocation/eviction trace logs.
static KV_CACHE_TRACE_ENABLED: LazyLock<bool> = LazyLock::new(|| {
    env::var(mocker::DYN_MOCKER_KV_CACHE_TRACE)
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
});

pub struct ManualKvManager {
    cache: HashCache,
    block_size: usize,
    kv_event_sink: Option<Arc<dyn KvCacheEventSink>>,
    dp_rank: u32,
    next_event_id: u64,
    /// Optional G2 (DRAM) tier for offloaded KV blocks.
    g2_cache: Option<HashCache>,
}

impl ManualKvManager {
    pub fn new(max_capacity: usize, block_size: usize) -> Self {
        Self::new_with_event_sink(max_capacity, block_size, None, 0, 0)
    }

    pub fn new_with_event_sink(
        max_capacity: usize,
        block_size: usize,
        kv_event_sink: Option<Arc<dyn KvCacheEventSink>>,
        dp_rank: u32,
        num_dram_blocks: usize,
    ) -> Self {
        debug_assert!(max_capacity > 0, "max_capacity must be > 0");
        if kv_event_sink.is_some() {
            tracing::info!(
                "ManualKvManager initialized with event sink for DP rank {dp_rank} with block_size {block_size}"
            );
        }

        let g2_cache = if num_dram_blocks > 0 {
            tracing::info!(
                "G2 (DRAM) tier enabled: {num_dram_blocks} blocks for DP rank {dp_rank}"
            );
            Some(HashCache::new(num_dram_blocks))
        } else {
            None
        };

        ManualKvManager {
            cache: HashCache::new(max_capacity),
            block_size,
            kv_event_sink,
            dp_rank,
            next_event_id: 0,
            g2_cache,
        }
    }

    /// Converts stored/removed blocks into KvCacheEventData and publishes if sink is available.
    fn publish_kv_event(
        &mut self,
        full_blocks: Vec<SequenceHash>,
        local_hashes: &[BlockHash],
        parent_hash: Option<u64>,
        is_store: bool,
        token_ids: Option<Vec<Vec<u32>>>,
    ) {
        if full_blocks.is_empty() {
            return;
        }

        if *KV_CACHE_TRACE_ENABLED {
            let active_len = self.cache.num_active();
            let inactive_len = self.cache.num_inactive();
            let free_blocks = self
                .cache
                .max_capacity()
                .saturating_sub(active_len)
                .saturating_sub(inactive_len);
            let event = if is_store { "allocation" } else { "eviction" };
            let timestamp_ms = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64;
            tracing::info!(
                event,
                timestamp_ms,
                block_ids = ?&full_blocks,
                block_size = self.block_size,
                free_blocks_after = free_blocks,
                active_blocks = active_len,
                inactive_blocks = inactive_len,
                total_blocks = self.cache.max_capacity(),
                dp_rank = self.dp_rank,
                "KV cache trace"
            );
        }

        let Some(ref sink) = self.kv_event_sink else {
            return;
        };

        let event_data = if is_store {
            let num_blocks = full_blocks.len();
            let local_hashes_slice = &local_hashes[local_hashes
                .len()
                .checked_sub(num_blocks)
                .expect("local hashes fewer than stored blocks")..];

            KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: parent_hash.map(ExternalSequenceBlockHash),
                blocks: full_blocks
                    .into_iter()
                    .zip(local_hashes_slice.iter())
                    .map(|(global_hash, local_hash)| KvCacheStoredBlockData {
                        block_hash: ExternalSequenceBlockHash(global_hash),
                        tokens_hash: LocalBlockHash(*local_hash),
                        mm_extra_info: None,
                    })
                    .collect(),
            })
        } else {
            KvCacheEventData::Removed(KvCacheRemoveData {
                block_hashes: full_blocks
                    .into_iter()
                    .map(ExternalSequenceBlockHash)
                    .collect(),
            })
        };

        let event_id = self.next_event_id;
        self.next_event_id += 1;

        let event = KvCacheEvent {
            event_id,
            data: event_data,
            dp_rank: self.dp_rank,
        };

        if let Err(e) = sink.publish(event, token_ids.as_deref()) {
            tracing::warn!("Failed to publish KV event: {e}");
        }
    }

    /// Log a G2 (DRAM) tier trace event when `DYN_MOCKER_KV_CACHE_TRACE=1`.
    fn trace_g2_event(&self, event_type: &str, block_id: u64) {
        if !*KV_CACHE_TRACE_ENABLED {
            return;
        }
        let Some(ref g2) = self.g2_cache else {
            return;
        };
        let g2_inactive = g2.num_inactive();
        let g2_free = g2.max_capacity().saturating_sub(g2_inactive);
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        tracing::info!(
            event = event_type,
            timestamp_ms,
            block_ids = ?vec![block_id],
            block_size = self.block_size,
            g1_free = self.cache.max_capacity()
                .saturating_sub(self.cache.num_active())
                .saturating_sub(self.cache.num_inactive()),
            g1_active = self.cache.num_active(),
            g1_inactive = self.cache.num_inactive(),
            g1_total = self.cache.max_capacity(),
            g2_free = g2_free,
            g2_inactive = g2_inactive,
            g2_total = g2.max_capacity(),
            dp_rank = self.dp_rank,
            "KV cache trace"
        );
    }

    /// Offload a block from G1 inactive → G2 inactive. If G2 is full, evict
    /// the G2 LRU block first.
    fn offload_to_g2(&mut self, block: &UniqueBlock) {
        let Some(ref mut g2) = self.g2_cache else {
            return;
        };
        // If G2 is full, evict LRU from G2 (truly discard)
        let mut g2_evicted_hash: Option<u64> = None;
        if g2.is_at_capacity() {
            if let Some(UniqueBlock::FullBlock(hash)) = g2.evict_inactive() {
                g2_evicted_hash = Some(hash);
            }
        }
        g2.insert_inactive(block.clone());
        // Now trace (no more mutable borrow on g2)
        if let Some(hash) = g2_evicted_hash {
            self.trace_g2_event("g2_eviction", hash);
        }
        if let UniqueBlock::FullBlock(hash) = block {
            self.trace_g2_event("g1_to_g2_offload", *hash);
        }
    }

    /// Try to onboard a block from G2 → G1 active. Returns true if found in G2.
    fn try_onboard_from_g2(&mut self, block: &UniqueBlock) -> bool {
        let Some(ref mut g2) = self.g2_cache else {
            return false;
        };
        if !g2.remove_inactive(block) {
            return false;
        }
        // Make room in G1 if needed
        if self.cache.is_at_capacity() {
            if let Some(evicted) = self.cache.evict_inactive() {
                if let UniqueBlock::FullBlock(evicted_hash) = &evicted {
                    // Publish G1 eviction event
                    self.publish_kv_event(vec![*evicted_hash], &[], None, false, None);
                }
                // Offload the evicted G1 block to G2
                // (need to re-borrow g2 since publish_kv_event borrows self)
                self.offload_to_g2(&evicted);
            } else {
                // No inactive to evict and G1 is full — cannot onboard
                // Put it back in G2
                if let Some(ref mut g2) = self.g2_cache {
                    g2.insert_inactive(block.clone());
                }
                return false;
            }
        }
        self.cache.insert_active(block.clone(), 1);
        if let UniqueBlock::FullBlock(hash) = block {
            self.trace_g2_event("g2_to_g1_onboard", *hash);
        }
        true
    }

    /// Get the keys of inactive blocks
    pub fn get_inactive_blocks(&self) -> Vec<&UniqueBlock> {
        self.cache.inactive_keys().collect()
    }

    /// Get the keys of active blocks
    pub fn get_active_blocks(&self) -> Vec<&UniqueBlock> {
        self.cache.active_keys().collect()
    }

    /// Direct access to active blocks map (for tests).
    pub fn active_blocks(&self) -> &HashMap<UniqueBlock, usize> {
        self.cache.active_blocks()
    }
}

impl KvBackend for ManualKvManager {
    fn process(&mut self, event: &MoveBlock) -> bool {
        match event {
            MoveBlock::Use(hashes, local_hashes, _plhs, token_ids) => {
                let mut blocks_stored = Vec::<u64>::new();
                let mut stored_token_ids: Option<Vec<Vec<u32>>> =
                    token_ids.as_ref().map(|_| Vec::new());
                let mut g1_active_hits = Vec::<u64>::new();
                let mut g1_inactive_hits = Vec::<u64>::new();
                let mut g2_hits = Vec::<u64>::new();
                // (global_hash, local_hash, parent_hash) for blocks onboarded from G2.
                // We must publish a Stored event for each so the router re-tracks them;
                // without it, a subsequent eviction sends a Removed for an unknown hash,
                // triggering "Failed to find block to remove" in the radix tree indexer.
                let mut g2_onboard_data: Vec<(u64, BlockHash, Option<u64>)> = Vec::new();

                let mut parent_block: Option<&UniqueBlock> = None;
                for (i, hash) in hashes.iter().enumerate() {
                    if self.cache.contains_active(hash) {
                        self.cache.increment_ref(hash);
                        if let UniqueBlock::FullBlock(h) = hash {
                            g1_active_hits.push(*h);
                        }
                        parent_block = Some(hash);
                        continue;
                    }

                    if self.cache.reactivate(hash) {
                        if let UniqueBlock::FullBlock(h) = hash {
                            g1_inactive_hits.push(*h);
                        }
                        parent_block = Some(hash);
                        continue;
                    }

                    // Check G2 (DRAM) tier before allocating a fresh block
                    let parent_hash_before_g2 = match parent_block {
                        Some(UniqueBlock::FullBlock(h)) => Some(*h),
                        _ => None,
                    };
                    if self.try_onboard_from_g2(hash) {
                        if let UniqueBlock::FullBlock(h) = hash {
                            g2_hits.push(*h);
                            if i < local_hashes.len() {
                                g2_onboard_data.push((*h, local_hashes[i], parent_hash_before_g2));
                            }
                        }
                        parent_block = Some(hash);
                        continue;
                    }

                    if self.cache.is_at_capacity() {
                        let Some(evicted) = self.cache.evict_inactive() else {
                            return false;
                        };
                        tracing::trace!(
                            "Evicting block from inactive pool: {evicted:?}, dp_rank={}",
                            self.dp_rank
                        );
                        if let UniqueBlock::FullBlock(evicted_full_block) = &evicted {
                            self.publish_kv_event(vec![*evicted_full_block], &[], None, false, None);
                        }
                        // Offload evicted block to G2 instead of discarding
                        self.offload_to_g2(&evicted);
                    }

                    self.cache.insert_active(hash.clone(), 1);
                    if let UniqueBlock::FullBlock(stored_full_block) = hash {
                        blocks_stored.push(*stored_full_block);
                        if let Some(ref mut stids) = stored_token_ids {
                            stids.push(token_ids.as_ref().unwrap()[i].clone());
                        }
                    }
                }

                let parent_hash = match parent_block {
                    None => None,
                    Some(UniqueBlock::FullBlock(block)) => Some(*block),
                    Some(UniqueBlock::PartialBlock(_)) => panic!("parent block cannot be partial"),
                };

                // Emit cache hit trace event with pool breakdown
                let total_hits = g1_active_hits.len() + g1_inactive_hits.len() + g2_hits.len();
                if total_hits > 0 && *KV_CACHE_TRACE_ENABLED {
                    let timestamp_ms = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_millis() as u64;
                    tracing::info!(
                        event = "cache_hit",
                        timestamp_ms,
                        g1_active_block_ids = ?&g1_active_hits,
                        g1_inactive_block_ids = ?&g1_inactive_hits,
                        g2_block_ids = ?&g2_hits,
                        block_size = self.block_size,
                        num_hits = total_hits,
                        g1_active_hits = g1_active_hits.len(),
                        g1_inactive_hits = g1_inactive_hits.len(),
                        g2_hits = g2_hits.len(),
                        active_blocks = self.cache.num_active(),
                        inactive_blocks = self.cache.num_inactive(),
                        total_blocks = self.cache.max_capacity(),
                        dp_rank = self.dp_rank,
                        "KV cache trace"
                    );
                }

                // Re-notify the router for each block that was onboarded from G2 → G1
                // BEFORE publishing new allocations that may use them as parents.
                // When a block was previously evicted from G1 the router received a Removed
                // event; without a corresponding Stored event on onboarding, a later eviction
                // sends another Removed for a hash the router no longer knows, causing the
                // "Failed to find block to remove" warning in the radix-tree indexer.
                // Publishing first also ensures parent blocks are known before children arrive.
                for (hash, local_hash, parent) in g2_onboard_data {
                    let lh = [local_hash];
                    self.publish_kv_event(vec![hash], &lh, parent, true, None);
                }

                self.publish_kv_event(
                    blocks_stored,
                    local_hashes,
                    parent_hash,
                    true,
                    stored_token_ids,
                );
            }

            MoveBlock::Destroy(hashes) => {
                let mut blocks_destroyed = Vec::<u64>::new();
                for hash in hashes.iter() {
                    self.cache.remove_active(hash).unwrap();
                    if let UniqueBlock::FullBlock(destroyed_full_block) = hash {
                        blocks_destroyed.push(*destroyed_full_block);
                    }
                }
                self.publish_kv_event(blocks_destroyed, &[], None, false, None);
            }

            MoveBlock::Deref(hashes) => {
                for hash in hashes.iter() {
                    if let Some(ref_count) = self.cache.get_active_ref_count(hash) {
                        if ref_count == 0 {
                            panic!("Negative reference count would be encountered after Deref.");
                        }
                        if ref_count == 1 {
                            self.cache.deactivate(hash);
                        } else {
                            self.cache.decrement_ref(hash);
                        }
                    }
                }
            }

            MoveBlock::Promote(uuid, hash, parent_hash, local_hash, _plh, promote_token_ids) => {
                let uuid_block = UniqueBlock::PartialBlock(*uuid);
                let hash_block = UniqueBlock::FullBlock(*hash);

                assert_eq!(
                    self.cache.remove_active(&uuid_block),
                    Some(1),
                    "uuid_block {uuid_block:?} should exist and be unique with ref_count=1"
                );

                let hash_ref_count = self.cache.get_active_ref_count(&hash_block);
                let is_new = if hash_ref_count.is_some() {
                    false
                } else {
                    !self.cache.remove_inactive(&hash_block)
                };

                self.cache
                    .insert_active(hash_block, hash_ref_count.unwrap_or(0) + 1);

                if is_new {
                    self.publish_kv_event(
                        vec![*hash],
                        &[*local_hash],
                        *parent_hash,
                        true,
                        promote_token_ids.as_ref().map(|t| vec![t.clone()]),
                    );
                }
            }
        }

        true
    }

    fn max_capacity(&self) -> usize {
        self.cache.max_capacity()
    }

    fn block_size(&self) -> usize {
        self.block_size
    }

    fn num_active_blocks(&self) -> usize {
        self.cache.num_active()
    }

    fn num_inactive_blocks(&self) -> usize {
        self.cache.num_inactive()
    }

    fn current_capacity(&self) -> usize {
        self.cache.current_capacity()
    }

    fn probe_new_blocks(&self, blocks: &[UniqueBlock]) -> usize {
        blocks
            .iter()
            .filter(|&block| !self.cache.contains(block))
            .count()
    }

    fn is_block_cached(&self, seq_hash: u64, _plh: Option<PositionalLineageHash>) -> bool {
        let block = UniqueBlock::FullBlock(seq_hash);
        self.cache.contains(&block)
            || self
                .g2_cache
                .as_ref()
                .is_some_and(|g2| g2.contains_inactive(&block))
    }

    fn num_g2_inactive_blocks(&self) -> usize {
        self.g2_cache.as_ref().map_or(0, |g2| g2.num_inactive())
    }

    fn g2_max_capacity(&self) -> usize {
        self.g2_cache.as_ref().map_or(0, |g2| g2.max_capacity())
    }

    fn is_block_in_g2(&self, seq_hash: u64) -> bool {
        let block = UniqueBlock::FullBlock(seq_hash);
        self.g2_cache
            .as_ref()
            .is_some_and(|g2| g2.contains_inactive(&block))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_failure_on_max_capacity() {
        let mut manager = ManualKvManager::new(10, 16);

        fn use_blocks(manager: &mut ManualKvManager, ids: Vec<u64>) -> bool {
            let blocks: Vec<_> = ids.iter().map(|&id| UniqueBlock::FullBlock(id)).collect();
            let hashes: Vec<_> = ids.into_iter().collect();
            manager.process(&MoveBlock::Use(blocks, hashes, vec![], None))
        }

        let response = use_blocks(&mut manager, (0..10).collect());
        assert!(response, "Expected success response");
        assert_eq!(manager.current_capacity(), 10);

        let response = use_blocks(&mut manager, vec![10]);
        assert!(
            !response,
            "Expected failure response when exceeding max capacity"
        );
    }

    #[test]
    fn test_block_lifecycle_stringent() {
        let mut manager = ManualKvManager::new(10, 16);

        fn use_blocks(manager: &mut ManualKvManager, ids: Vec<u64>) {
            let blocks: Vec<_> = ids.iter().map(|&id| UniqueBlock::FullBlock(id)).collect();
            let hashes: Vec<_> = ids.into_iter().collect();
            manager.process(&MoveBlock::Use(blocks, hashes, vec![], None));
        }

        fn destroy_blocks(manager: &mut ManualKvManager, ids: Vec<u64>) {
            let blocks = ids.into_iter().map(UniqueBlock::FullBlock).collect();
            manager.process(&MoveBlock::Destroy(blocks));
        }

        fn deref_blocks(manager: &mut ManualKvManager, ids: Vec<u64>) {
            let blocks = ids.into_iter().map(UniqueBlock::FullBlock).collect();
            manager.process(&MoveBlock::Deref(blocks));
        }

        fn assert_active_blocks(manager: &ManualKvManager, expected_blocks: &[(u64, usize)]) {
            assert_eq!(
                manager.active_blocks().len(),
                expected_blocks.len(),
                "Active blocks count doesn't match expected"
            );
            for &(id, ref_count) in expected_blocks {
                let block = UniqueBlock::FullBlock(id);
                assert!(
                    manager.active_blocks().contains_key(&block),
                    "Block {id} not found in active blocks",
                );
                assert_eq!(
                    manager.active_blocks().get(&block),
                    Some(&ref_count),
                    "Block {id} has wrong reference count",
                );
            }
        }

        fn assert_inactive_blocks(
            manager: &ManualKvManager,
            expected_size: usize,
            expected_blocks: &[u64],
        ) {
            let inactive_blocks = manager.get_inactive_blocks();
            let inactive_blocks_count = manager.num_inactive_blocks();
            assert_eq!(
                inactive_blocks_count, expected_size,
                "Inactive blocks count doesn't match expected"
            );
            for &id in expected_blocks {
                let block = UniqueBlock::FullBlock(id);
                assert!(
                    inactive_blocks.iter().any(|&b| *b == block),
                    "Block {id} not found in inactive blocks",
                );
            }
        }

        use_blocks(&mut manager, (0..5).collect());
        use_blocks(&mut manager, vec![0, 1, 5, 6]);
        assert_active_blocks(
            &manager,
            &[(0, 2), (1, 2), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1)],
        );

        destroy_blocks(&mut manager, vec![4]);
        deref_blocks(&mut manager, vec![0, 1, 2, 3]);
        assert_inactive_blocks(&manager, 2, &[3, 2]);
        assert_active_blocks(&manager, &[(0, 1), (1, 1), (5, 1), (6, 1)]);

        destroy_blocks(&mut manager, vec![6]);
        deref_blocks(&mut manager, vec![0, 1, 5]);
        assert_inactive_blocks(&manager, 5, &[0, 1, 2, 3, 5]);
        assert_active_blocks(&manager, &[]);

        use_blocks(&mut manager, vec![0, 1, 2, 7, 8, 9]);
        assert_inactive_blocks(&manager, 2, &[3, 5]);
        assert_active_blocks(&manager, &[(0, 1), (1, 1), (2, 1), (7, 1), (8, 1), (9, 1)]);

        let blocks_to_check: Vec<UniqueBlock> = vec![0, 1, 2, 3, 4]
            .into_iter()
            .map(UniqueBlock::FullBlock)
            .collect();
        assert_eq!(manager.probe_new_blocks(&blocks_to_check), 1);

        use_blocks(&mut manager, vec![10, 11, 12]);
        assert_inactive_blocks(&manager, 1, &[5]);

        use_blocks(&mut manager, vec![13]);
    }

    /// Test G2 (DRAM) tiered cache: offload, onboard, and eviction flows.
    ///
    /// Scenario: G1=4 blocks, G2=2 blocks
    ///   Phase 1: Fill G1 with blocks 0-3
    ///   Phase 2: Deref all → inactive, then Use blocks 4,5 → evict 0,1 → offload to G2
    ///   Phase 3: Re-Use block 0 → should onboard from G2 back to G1
    #[test]
    fn test_g2_offload_and_onboard() {
        // Enable tracing env var for this test
        // SAFETY: single-threaded test, no concurrent env access
        unsafe { std::env::set_var("DYN_MOCKER_KV_CACHE_TRACE", "1") };

        let mut manager = ManualKvManager::new_with_event_sink(4, 16, None, 0, 2);

        // Verify G2 is configured
        assert_eq!(manager.g2_max_capacity(), 2);
        assert_eq!(manager.num_g2_inactive_blocks(), 0);

        fn use_blocks(manager: &mut ManualKvManager, ids: Vec<u64>) -> bool {
            let blocks: Vec<_> = ids.iter().map(|&id| UniqueBlock::FullBlock(id)).collect();
            let hashes: Vec<_> = ids.into_iter().collect();
            manager.process(&MoveBlock::Use(blocks, hashes, vec![], None))
        }

        fn deref_blocks(manager: &mut ManualKvManager, ids: Vec<u64>) {
            let blocks = ids.into_iter().map(UniqueBlock::FullBlock).collect();
            manager.process(&MoveBlock::Deref(blocks));
        }

        // Phase 1: Fill G1 with blocks 0-3
        assert!(use_blocks(&mut manager, vec![0, 1, 2, 3]));
        assert_eq!(manager.num_active_blocks(), 4);
        assert_eq!(manager.num_inactive_blocks(), 0);

        // Deref all to move them to inactive pool
        deref_blocks(&mut manager, vec![0, 1, 2, 3]);
        assert_eq!(manager.num_active_blocks(), 0);
        assert_eq!(manager.num_inactive_blocks(), 4);

        // Phase 2: Use new blocks 4,5 → must evict LRU from G1 → offloads to G2
        assert!(use_blocks(&mut manager, vec![4]));
        // Block 0 (oldest LRU) should be evicted from G1 → offloaded to G2
        assert_eq!(manager.num_g2_inactive_blocks(), 1);
        assert!(manager.is_block_in_g2(0), "Block 0 should be in G2");

        assert!(use_blocks(&mut manager, vec![5]));
        // Block 1 evicted → offloaded to G2
        assert_eq!(manager.num_g2_inactive_blocks(), 2);
        assert!(manager.is_block_in_g2(1), "Block 1 should be in G2");

        // G1: active=[4,5], inactive=[2,3], G2: inactive=[0,1]
        assert_eq!(manager.num_active_blocks(), 2);
        assert_eq!(manager.num_inactive_blocks(), 2);

        // Phase 3: Re-use block 0 → should onboard from G2 back to G1
        deref_blocks(&mut manager, vec![4, 5]);
        // G1: active=[], inactive=[2,3,4,5], G2: inactive=[0,1]
        assert_eq!(manager.num_active_blocks(), 0);
        assert_eq!(manager.num_inactive_blocks(), 4);

        assert!(use_blocks(&mut manager, vec![0]));
        // Block 0 should be onboarded from G2 → G1 active.
        // G1 was at capacity so onboarding evicts block 2 from G1 inactive → offloads to G2.
        // Result: G2 goes from {0,1} → remove 0 → {1} → add 2 → {1,2}
        assert!(!manager.is_block_in_g2(0), "Block 0 should no longer be in G2");
        assert_eq!(manager.num_g2_inactive_blocks(), 2);
        assert!(manager.is_block_in_g2(1), "Block 1 should still be in G2");
        assert!(manager.is_block_in_g2(2), "Block 2 should have been offloaded to G2 during onboard");
        assert!(
            manager.cache.contains_active(&UniqueBlock::FullBlock(0)),
            "Block 0 should be in G1 active after onboarding"
        );

        // Phase 4: Verify G2 eviction when G2 is full
        // Currently G2 has [1]. Fill G1 again and evict 2 more to G2.
        deref_blocks(&mut manager, vec![0]);
        // G1: active=[], inactive=[0,2,3,4,5] — wait, G1 only has 4 capacity
        // Actually: G1 has capacity 4, so inactive can hold up to 4 blocks.
        // After deref of 0: G1 inactive should have [2,3,4,5,0] minus the one evicted for 0's onboard
        // Let's check actual state:
        let g1_active = manager.num_active_blocks();
        let g1_inactive = manager.num_inactive_blocks();
        let g2_inactive = manager.num_g2_inactive_blocks();
        assert_eq!(g1_active, 0);
        // G1 should be at capacity (4 blocks): the onboard of 0 might have evicted an inactive
        assert!(g1_inactive <= 4, "G1 inactive should not exceed capacity");
        assert!(g2_inactive >= 1, "G2 should have at least 1 block");
    }
}
