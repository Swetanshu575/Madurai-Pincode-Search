# planner.py
import time
import numpy as np
from typing import Dict, Tuple
from ivf_index import CustomIVFIndex, FilterSpec


class SelectivityEstimator:
    def __init__(self, index: CustomIVFIndex, num_price_bins: int = 100, seed: int = 42):
        self.num_price_bins = num_price_bins
        self.seed = seed
        self.refresh_stats(index)

    def refresh_stats(self, index: CustomIVFIndex):
        alive_mask = index.alive
        self.total_alive = int(np.sum(alive_mask))
        if self.total_alive == 0:
            return

        active_cats = index.cat_ids[alive_mask]
        active_zones = index.zone_ids[alive_mask]
        active_prices = index.prices[alive_mask]

        # 1. hash maps for category and zone counts
        u_c, cnt_c = np.unique(active_cats, return_counts=True)
        self.cat_counts: Dict[int, int] = dict(zip(map(int, u_c), map(int, cnt_c)))

        u_z, cnt_z = np.unique(active_zones, return_counts=True)
        self.zone_counts: Dict[int, int] = dict(zip(map(int, u_z), map(int, cnt_z)))

        # 2. log-scale price histogram for cdf interpolation
        self.min_p = float(np.min(active_prices))
        self.max_p = float(np.max(active_prices))
        self.bin_edges = np.geomspace(max(1.0, self.min_p), self.max_p + 1.0, self.num_price_bins + 1)
        cnt_p, _ = np.histogram(active_prices, bins=self.bin_edges)
        self.price_cdf = np.concatenate([[0.0], np.cumsum(cnt_p) / self.total_alive])

        # 3. 1% micro-sample sketch (500 items) to catch joint correlation breakdowns
        rng = np.random.default_rng(self.seed)
        sample_size = min(500, self.total_alive)
        sample_idx = rng.choice(np.flatnonzero(alive_mask), size=sample_size, replace=False)
        self.sample_cats = index.cat_ids[sample_idx]
        self.sample_zones = index.zone_ids[sample_idx]
        self.sample_prices = index.prices[sample_idx]
        self.sample_size = sample_size

    def _estimate_price_prob(self, min_price: float = None, max_price: float = None) -> float:
        lo = self.min_p if min_price is None else max(self.min_p, min_price)
        hi = self.max_p if max_price is None else min(self.max_p, max_price)
        if hi < lo:
            return 0.0
        cdf_lo = float(np.interp(lo, self.bin_edges, self.price_cdf))
        cdf_hi = float(np.interp(hi, self.bin_edges, self.price_cdf))
        return max(0.0, cdf_hi - cdf_lo)

    def estimate_independent(self, f: FilterSpec) -> float:
        # assumes predicates are independent: P(cat) * P(zone) * P(price)
        if self.total_alive == 0:
            return 0.0
        s_hat = 1.0
        if f.cat_id is not None:
            s_hat *= self.cat_counts.get(int(f.cat_id), 0) / self.total_alive
        if f.zone_id is not None:
            s_hat *= self.zone_counts.get(int(f.zone_id), 0) / self.total_alive
        if f.min_price is not None or f.max_price is not None:
            s_hat *= self._estimate_price_prob(f.min_price, f.max_price)
        return float(np.clip(s_hat, 0.0, 1.0))

    def estimate_with_sketch(self, f: FilterSpec) -> float:
        # check micro-sample when multiple predicates are combined
        ind_est = self.estimate_independent(f)
        num_preds = sum([
            f.cat_id is not None,
            f.zone_id is not None,
            (f.min_price is not None or f.max_price is not None),
        ])
        if num_preds <= 1:
            return ind_est

        m = np.ones(self.sample_size, dtype=bool)
        if f.cat_id is not None:
            m &= (self.sample_cats == f.cat_id)
        if f.zone_id is not None:
            m &= (self.sample_zones == f.zone_id)
        if f.min_price is not None:
            m &= (self.sample_prices >= f.min_price)
        if f.max_price is not None:
            m &= (self.sample_prices <= f.max_price)

        hits = int(np.sum(m))
        # if independent estimate expects >0.8% (~4+ hits in 500) but sample has 0 hits,
        # joint correlation has collapsed the subspace
        if ind_est > 0.008 and hits == 0:
            return 1.0 / self.total_alive
        if hits >= 3:
            return hits / self.sample_size
        return ind_est


class CostBasedQueryPlanner:
    STRATEGY_EXACT = "EXACT_PREFILTER"
    STRATEGY_RESTRICTED = "RESTRICTED_IVF"
    STRATEGY_ADAPTIVE_POST = "ADAPTIVE_POST_IVF"

    def __init__(
        self,
        index: CustomIVFIndex,
        estimator: SelectivityEstimator,
        tau_low: float = 0.025,
        tau_high: float = 0.30,
        use_sketch_fix: bool = False,
    ):
        self.index = index
        self.estimator = estimator
        self.tau_low = tau_low
        self.tau_high = tau_high
        self.use_sketch_fix = use_sketch_fix

    def calibrate_thresholds(self, sample_queries: np.ndarray) -> Tuple[float, float]:
        # benchmark exact scan vs restricted ivf across selectivity points
        selectivities = [0.002, 0.005, 0.01, 0.02, 0.035, 0.05, 0.10, 0.25, 0.50]
        crossover_s = 0.025

        for s in selectivities:
            p_cutoff = float(np.quantile(self.index.prices[self.index.alive], s))
            f = FilterSpec(max_price=p_cutoff)

            # time exact scan
            t0 = time.perf_counter()
            for q in sample_queries[:25]:
                self.index.search_exact_prefilter(q, f, k=10)
            t_exact = (time.perf_counter() - t0) / 25

            # time ivf
            nprobe = self._compute_adaptive_nprobe(s)
            t0 = time.perf_counter()
            for q in sample_queries[:25]:
                self.index.search_restricted_ivf(q, f, k=10, nprobe=nprobe)
            t_ivf = (time.perf_counter() - t0) / 25

            if t_ivf < t_exact:
                crossover_s = s
                break

        self.tau_low = float(np.clip(crossover_s, 0.012, 0.040))
        self.tau_high = 0.30
        return self.tau_low, self.tau_high

    def _compute_adaptive_nprobe(self, s_hat: float) -> int:
        if s_hat >= 0.50:
            return 22
        elif s_hat >= 0.20:
            return 28
        elif s_hat >= 0.08:
            return 38
        elif s_hat >= 0.03:
            return 52
        else:
            return 72

    def choose_strategy(self, f: FilterSpec) -> Tuple[str, float, int]:
        s_hat = (
            self.estimator.estimate_with_sketch(f)
            if self.use_sketch_fix
            else self.estimator.estimate_independent(f)
        )
        nprobe = self._compute_adaptive_nprobe(max(s_hat, 1e-4))

        if s_hat <= self.tau_low:
            return self.STRATEGY_EXACT, s_hat, 0
        elif s_hat <= self.tau_high:
            return self.STRATEGY_RESTRICTED, s_hat, nprobe
        else:
            return self.STRATEGY_ADAPTIVE_POST, s_hat, nprobe

    def search(self, query_vec: np.ndarray, f: FilterSpec, k: int = 10):
        strategy, s_hat, nprobe = self.choose_strategy(f)

        if strategy == self.STRATEGY_EXACT:
            ids, scores = self.index.search_exact_prefilter(query_vec, f, k=k)
        elif strategy == self.STRATEGY_RESTRICTED:
            ids, scores = self.index.search_restricted_ivf(
                query_vec, f, k=k, nprobe=nprobe, min_candidates=750
            )
        else:
            ids, scores = self.index.search_adaptive_postfilter_ivf(
                query_vec, f, k=k, nprobe=nprobe
            )

        return ids, scores, strategy, s_hat