# baselines.py
import numpy as np
from ivf_index import CustomIVFIndex, FilterSpec


def exact_ground_truth(index: CustomIVFIndex, query_vec: np.ndarray, f: FilterSpec, k: int = 10):
    # exact brute force scan over matching items
    return index.search_exact_prefilter(query_vec, f, k=k)


class PurePostFilterBaseline:
    # baseline 1: standard ann search first (top 100), then throw away non-matching
    def __init__(self, index: CustomIVFIndex, top_m: int = 100, fixed_nprobe: int = 16):
        self.index = index
        self.top_m = top_m
        self.fixed_nprobe = fixed_nprobe

    def search(self, query_vec: np.ndarray, f: FilterSpec, k: int = 10):
        # find top nprobe centroids
        c_scores = self.index.centroids @ query_vec
        nprobe = min(self.fixed_nprobe, len(self.index.centroids))
        top_cids = np.argpartition(-c_scores, nprobe - 1)[:nprobe]

        cand_ids = []
        cand_scores = []

        for cid in top_cids:
            part = self.index.partitions[cid]
            if part.active_count == 0:
                continue
            alive_idx = np.flatnonzero(part.alive)
            if len(alive_idx) == 0:
                continue
            scores = part.vectors[alive_idx] @ query_vec
            cand_ids.append(part.item_ids[alive_idx])
            cand_scores.append(scores)

        if not cand_ids:
            return np.array([], dtype=np.int32), np.array([], dtype=np.float32)

        all_ids = np.concatenate(cand_ids)
        all_scores = np.concatenate(cand_scores)

        # take global top M candidates first
        m = min(self.top_m, len(all_scores))
        top_m_idx = np.argpartition(-all_scores, m - 1)[:m]
        top_m_idx = top_m_idx[np.argsort(-all_scores[top_m_idx])]

        survivor_ids = all_ids[top_m_idx]
        survivor_scores = all_scores[top_m_idx]

        # filter survivors against requested predicates
        mask = self.index.global_filter_mask(f)
        valid_set = set(self.index.item_ids[mask])

        keep = [i for i, item_id in enumerate(survivor_ids) if int(item_id) in valid_set][:k]
        if not keep:
            return np.array([], dtype=np.int32), np.array([], dtype=np.float32)

        keep_idx = np.array(keep, dtype=np.int32)
        return survivor_ids[keep_idx], survivor_scores[keep_idx]


class PurePreFilterBaseline:
    # baseline 2: filter dataset first, then brute-force scan survivors
    def __init__(self, index: CustomIVFIndex):
        self.index = index

    def search(self, query_vec: np.ndarray, f: FilterSpec, k: int = 10):
        return self.index.search_exact_prefilter(query_vec, f, k=k)