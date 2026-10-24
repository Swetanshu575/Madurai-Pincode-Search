# ivf_index.py
import time
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from sklearn.cluster import MiniBatchKMeans


@dataclass
class FilterSpec:
    cat_id: Optional[int] = None
    zone_id: Optional[int] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None

    def describe(self, categories: List[str], zones: List[str]) -> str:
        parts = []
        if self.cat_id is not None:
            name = categories[self.cat_id] if self.cat_id < len(categories) else f"cat_{self.cat_id}"
            parts.append(f"cat={name}")
        if self.zone_id is not None:
            zname = zones[self.zone_id] if self.zone_id < len(zones) else f"zone_{self.zone_id}"
            parts.append(f"zone={zname}")
        if self.min_price is not None or self.max_price is not None:
            lo = f"Rs.{int(self.min_price)}" if self.min_price is not None else "0"
            hi = f"Rs.{int(self.max_price)}" if self.max_price is not None else "inf"
            parts.append(f"price=[{lo},{hi}]")
        return " AND ".join(parts) if parts else "ALL"


class PartitionBlock:
    # keeps vectors and meta for a single IVF cluster, plus fast summary bitmasks
    def __init__(
        self,
        cluster_id: int,
        vectors: np.ndarray,
        item_ids: np.ndarray,
        cat_ids: np.ndarray,
        zone_ids: np.ndarray,
        prices: np.ndarray,
    ):
        self.cluster_id = cluster_id
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.item_ids = np.ascontiguousarray(item_ids, dtype=np.int32)
        self.cat_ids = np.ascontiguousarray(cat_ids, dtype=np.int16)
        self.zone_ids = np.ascontiguousarray(zone_ids, dtype=np.int16)
        self.prices = np.ascontiguousarray(prices, dtype=np.float32)
        # 1 = active, 0 = deleted
        self.alive = np.ones(len(item_ids), dtype=bool)
        self._rebuild_summary()

    def _rebuild_summary(self):
        active = self.alive
        self.active_count = int(np.sum(active))
        if self.active_count == 0:
            self.cat_bitmask = 0
            self.zone_bitmask = 0
            self.min_price = float("inf")
            self.max_price = float("-inf")
            return

        c_mask = 0
        for c in np.unique(self.cat_ids[active]):
            c_mask |= (1 << int(c))
        self.cat_bitmask = c_mask

        z_mask = 0
        for z in np.unique(self.zone_ids[active]):
            z_mask |= (1 << int(z))
        self.zone_bitmask = z_mask

        p = self.prices[active]
        self.min_price = float(np.min(p))
        self.max_price = float(np.max(p))

    def may_contain(self, f: FilterSpec) -> bool:
        # quick partition-level check using bitmasks before reading array rows
        if self.active_count == 0:
            return False
        if f.cat_id is not None and not (self.cat_bitmask & (1 << int(f.cat_id))):
            return False
        if f.zone_id is not None and not (self.zone_bitmask & (1 << int(f.zone_id))):
            return False
        if f.min_price is not None and self.max_price < f.min_price:
            return False
        if f.max_price is not None and self.min_price > f.max_price:
            return False
        return True

    def local_filter_mask(self, f: FilterSpec) -> np.ndarray:
        m = self.alive.copy()
        if f.cat_id is not None:
            m &= (self.cat_ids == f.cat_id)
        if f.zone_id is not None:
            m &= (self.zone_ids == f.zone_id)
        if f.min_price is not None:
            m &= (self.prices >= f.min_price)
        if f.max_price is not None:
            m &= (self.prices <= f.max_price)
        return m


class CustomIVFIndex:
    def __init__(self, n_list: int = 256, random_state: int = 42):
        self.n_list = n_list
        self.random_state = random_state
        self.centroids: Optional[np.ndarray] = None
        self.partitions: List[PartitionBlock] = []

        # global column storage for fast exact scanning
        self.vectors: Optional[np.ndarray] = None
        self.item_ids: Optional[np.ndarray] = None
        self.cat_ids: Optional[np.ndarray] = None
        self.zone_ids: Optional[np.ndarray] = None
        self.prices: Optional[np.ndarray] = None
        self.alive: Optional[np.ndarray] = None

    def build(self, vectors, item_ids, cat_ids, zone_ids, prices) -> float:
        t0 = time.time()
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.item_ids = np.ascontiguousarray(item_ids, dtype=np.int32)
        self.cat_ids = np.ascontiguousarray(cat_ids, dtype=np.int16)
        self.zone_ids = np.ascontiguousarray(zone_ids, dtype=np.int16)
        self.prices = np.ascontiguousarray(prices, dtype=np.float32)
        self.alive = np.ones(len(item_ids), dtype=bool)

        # kmeans to find cluster centers
        km = MiniBatchKMeans(
            n_clusters=self.n_list,
            batch_size=4096,
            n_init=3,
            max_iter=100,
            random_state=self.random_state,
        )
        labels = km.fit_predict(self.vectors)
        raw_c = km.cluster_centers_.astype(np.float32)
        c_norms = np.linalg.norm(raw_c, axis=1, keepdims=True)
        self.centroids = raw_c / np.maximum(c_norms, 1e-12)

        # create partition blocks
        self.partitions = []
        for cid in range(self.n_list):
            idx = np.flatnonzero(labels == cid)
            block = PartitionBlock(
                cluster_id=cid,
                vectors=self.vectors[idx],
                item_ids=self.item_ids[idx],
                cat_ids=self.cat_ids[idx],
                zone_ids=self.zone_ids[idx],
                prices=self.prices[idx],
            )
            self.partitions.append(block)

        return time.time() - t0

    def global_filter_mask(self, f: FilterSpec) -> np.ndarray:
        m = self.alive.copy()
        if f.cat_id is not None:
            m &= (self.cat_ids == f.cat_id)
        if f.zone_id is not None:
            m &= (self.zone_ids == f.zone_id)
        if f.min_price is not None:
            m &= (self.prices >= f.min_price)
        if f.max_price is not None:
            m &= (self.prices <= f.max_price)
        return m

    # Strategy 1: Exact pre-filter scan
    def search_exact_prefilter(self, query_vec: np.ndarray, f: FilterSpec, k: int = 10):
        mask = self.global_filter_mask(f)
        valid_idx = np.flatnonzero(mask)
        if len(valid_idx) == 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.float32)

        sub_vecs = self.vectors[valid_idx]
        scores = sub_vecs @ query_vec

        if len(scores) <= k:
            order = np.argsort(-scores)
        else:
            top_k = np.argpartition(-scores, k)[:k]
            order = top_k[np.argsort(-scores[top_k])]

        return self.item_ids[valid_idx[order]], scores[order]

    # Strategy 2: Partition-restricted IVF with early exit once min_candidates are found
    def search_restricted_ivf(
        self,
        query_vec: np.ndarray,
        f: FilterSpec,
        k: int = 10,
        nprobe: int = 16,
        min_candidates: int = 600,
    ):
        # filter out partitions that definitely have no matches
        qualifying = [p.cluster_id for p in self.partitions if p.may_contain(f)]
        if not qualifying:
            return np.array([], dtype=np.int32), np.array([], dtype=np.float32)

        qualifying_arr = np.array(qualifying, dtype=np.int32)
        c_scores = self.centroids[qualifying_arr] @ query_vec
        order = np.argsort(-c_scores)
        ranked_cids = qualifying_arr[order]

        cand_ids = []
        cand_scores = []
        seen = 0

        for i, cid in enumerate(ranked_cids):
            if i >= nprobe and seen >= min_candidates:
                break

            part = self.partitions[cid]
            loc_mask = part.local_filter_mask(f)
            loc_idx = np.flatnonzero(loc_mask)
            if len(loc_idx) == 0:
                continue

            sub = part.vectors[loc_idx]
            scores = sub @ query_vec
            cand_ids.append(part.item_ids[loc_idx])
            cand_scores.append(scores)
            seen += len(loc_idx)

        if not cand_ids:
            return np.array([], dtype=np.int32), np.array([], dtype=np.float32)

        all_ids = np.concatenate(cand_ids)
        all_scores = np.concatenate(cand_scores)

        if len(all_scores) <= k:
            ord_k = np.argsort(-all_scores)
        else:
            top_k = np.argpartition(-all_scores, k)[:k]
            ord_k = top_k[np.argsort(-all_scores[top_k])]

        return all_ids[ord_k], all_scores[ord_k]

    # Strategy 3: Adaptive post-filtering IVF for high selectivity queries
    def search_adaptive_postfilter_ivf(
        self, query_vec: np.ndarray, f: FilterSpec, k: int = 10, nprobe: int = 16
    ):
        c_scores = self.centroids @ query_vec
        nprobe = min(max(1, nprobe), self.n_list)
        top_cids = np.argpartition(-c_scores, nprobe - 1)[:nprobe]
        top_cids = top_cids[np.argsort(-c_scores[top_cids])]

        cand_ids = []
        cand_scores = []

        for cid in top_cids:
            part = self.partitions[cid]
            if part.active_count == 0:
                continue
            scores = part.vectors @ query_vec
            mask = part.local_filter_mask(f)
            if np.any(mask):
                cand_ids.append(part.item_ids[mask])
                cand_scores.append(scores[mask])

        if not cand_ids:
            return np.array([], dtype=np.int32), np.array([], dtype=np.float32)

        all_ids = np.concatenate(cand_ids)
        all_scores = np.concatenate(cand_scores)

        if len(all_scores) <= k:
            ord_k = np.argsort(-all_scores)
        else:
            top_k = np.argpartition(-all_scores, k)[:k]
            ord_k = top_k[np.argsort(-all_scores[top_k])]

        return all_ids[ord_k], all_scores[ord_k]

    # Live mutations: deletes & inserts
    def delete_items(self, item_ids_to_delete: np.ndarray) -> int:
        del_set = set(map(int, item_ids_to_delete))
        global_mask = np.isin(self.item_ids, list(del_set))
        del_count = int(np.sum(self.alive & global_mask))
        self.alive &= ~global_mask

        for part in self.partitions:
            p_del = np.isin(part.item_ids, list(del_set))
            if np.any(p_del):
                part.alive &= ~p_del
                part._rebuild_summary()

        return del_count

    def insert_items(self, new_vectors, new_item_ids, new_cat_ids, new_zone_ids, new_prices):
        new_vectors = np.ascontiguousarray(new_vectors, dtype=np.float32)

        # append to global buffers
        self.vectors = np.vstack([self.vectors, new_vectors])
        self.item_ids = np.concatenate([self.item_ids, new_item_ids.astype(np.int32)])
        self.cat_ids = np.concatenate([self.cat_ids, new_cat_ids.astype(np.int16)])
        self.zone_ids = np.concatenate([self.zone_ids, new_zone_ids.astype(np.int16)])
        self.prices = np.concatenate([self.prices, new_prices.astype(np.float32)])
        self.alive = np.concatenate([self.alive, np.ones(len(new_item_ids), dtype=bool)])

        # route to nearest centroid
        sims = new_vectors @ self.centroids.T
        assigned = np.argmax(sims, axis=1)

        for cid in np.unique(assigned):
            idx = np.flatnonzero(assigned == cid)
            part = self.partitions[int(cid)]
            part.vectors = np.ascontiguousarray(np.vstack([part.vectors, new_vectors[idx]]))
            part.item_ids = np.concatenate([part.item_ids, new_item_ids[idx].astype(np.int32)])
            part.cat_ids = np.concatenate([part.cat_ids, new_cat_ids[idx].astype(np.int16)])
            part.zone_ids = np.concatenate([part.zone_ids, new_zone_ids[idx].astype(np.int16)])
            part.prices = np.concatenate([part.prices, new_prices[idx].astype(np.float32)])
            part.alive = np.concatenate([part.alive, np.ones(len(idx), dtype=bool)])
            part._rebuild_summary()

        return assigned