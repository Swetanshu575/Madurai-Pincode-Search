# benchmark.py
import time
import numpy as np
from data_prep import load_or_create_dataset, CatalogueDataset
from ivf_index import CustomIVFIndex, FilterSpec
from baselines import exact_ground_truth, PurePostFilterBaseline, PurePreFilterBaseline
from planner import SelectivityEstimator, CostBasedQueryPlanner


def compute_recall_at_k(pred_ids, gt_ids, k=10):
    if len(gt_ids) == 0:
        return 1.0
    top_k = min(k, len(gt_ids))
    gt_set = set(gt_ids[:top_k])
    pred_set = set(pred_ids[:top_k])
    return len(gt_set & pred_set) / float(top_k)


def build_bucket_workload(ds: CatalogueDataset, index: CustomIVFIndex, n_queries=80, seed=101):
    rng = np.random.default_rng(seed)

    def sample_query(f: FilterSpec):
        mask = index.global_filter_mask(f)
        valid = np.flatnonzero(mask)
        if len(valid) > 0:
            idx = rng.choice(valid)
            base = index.vectors[idx]
        else:
            base = index.vectors[rng.integers(0, len(index.vectors))]
        # add a small amount of perturbation noise
        q = base + rng.normal(0, 0.14, size=base.shape).astype(np.float32)
        norm = np.linalg.norm(q)
        return (q / (norm + 1e-12)).astype(np.float32)

    buckets = {
        "1. ~100% (Broad/Mumbai)": [],
        "2. ~10%  (Metro/Cat)": [],
        "3. ~1%   (Tier-2/Combo)": [],
        "4. ~0.1% (Madurai Case)": [],
        "5. Handful (5-25 items)": [],
    }

    # bucket 1: broad searches (no filter or price cap under 85k)
    for _ in range(n_queries):
        f = FilterSpec(max_price=85000.0) if rng.random() < 0.5 else FilterSpec()
        buckets["1. ~100% (Broad/Mumbai)"].append((sample_query(f), f))

    # bucket 2: big hubs and popular categories (~10%)
    b2_filters = [
        FilterSpec(zone_id=ds.zone_to_id["Kolkata_Hub"]),
        FilterSpec(zone_id=ds.zone_to_id["Bengaluru_Hosur"]),
        FilterSpec(zone_id=ds.zone_to_id["Mumbai_Bhiwandi"], max_price=2200.0),
        FilterSpec(cat_id=ds.cat_to_id["kurta_set"]),
    ]
    for i in range(n_queries):
        f = b2_filters[i % len(b2_filters)]
        buckets["2. ~10%  (Metro/Cat)"].append((sample_query(f), f))

    # bucket 3: smaller hubs / joint category + zone (~1%)
    b3_filters = [
        FilterSpec(zone_id=ds.zone_to_id["Indore_Hub"]),
        FilterSpec(zone_id=ds.zone_to_id["Nagpur_Central"]),
        FilterSpec(zone_id=ds.zone_to_id["Kanpur_Zone"]),
        FilterSpec(zone_id=ds.zone_to_id["Mumbai_Bhiwandi"], cat_id=ds.cat_to_id["cotton_saree"]),
    ]
    for i in range(n_queries):
        f = b3_filters[i % len(b3_filters)]
        buckets["3. ~1%   (Tier-2/Combo)"].append((sample_query(f), f))

    # bucket 4: madurai saree scenario (~0.1% selectivity)
    b4_filters = [
        FilterSpec(
            zone_id=ds.zone_to_id["Madurai_TN"],
            cat_id=ds.cat_to_id["cotton_saree"],
            max_price=1500.0,
        ),
        FilterSpec(zone_id=ds.zone_to_id["Siliguri_WB"]),
        FilterSpec(zone_id=ds.zone_to_id["Tinsukia_AS"]),
        FilterSpec(
            zone_id=ds.zone_to_id["Coimbatore_TN"],
            cat_id=ds.cat_to_id["silk_saree"],
            max_price=6000.0,
        ),
    ]
    for i in range(n_queries):
        f = b4_filters[i % len(b4_filters)]
        buckets["4. ~0.1% (Madurai Case)"].append((sample_query(f), f))

    # bucket 5: very rare combos / niche pins (5 to 25 items)
    b5_filters = [
        FilterSpec(zone_id=ds.zone_to_id["Kargil_LK"], max_price=2500.0),
        FilterSpec(zone_id=ds.zone_to_id["PortBlair_AN"], max_price=2000.0),
        FilterSpec(
            zone_id=ds.zone_to_id["Madurai_TN"],
            cat_id=ds.cat_to_id["silk_saree"],
            max_price=4500.0,
        ),
        FilterSpec(
            zone_id=ds.zone_to_id["Ranchi_JH"],
            cat_id=ds.cat_to_id["smartphones"],
            max_price=18000.0,
        ),
    ]
    for i in range(n_queries):
        f = b5_filters[i % len(b5_filters)]
        buckets["5. Handful (5-25 items)"].append((sample_query(f), f))

    return list(buckets.items())


def run_benchmark():
    print("-----------------------------------------------------------------------------------------")
    print(" Bharat Bazaar - Madurai Pincode Search Benchmark")
    print("-----------------------------------------------------------------------------------------")

    ds = load_or_create_dataset(n_items=50_000)
    index = CustomIVFIndex(n_list=256, random_state=42)
    t0 = time.time()
    index.build(ds.vectors, ds.item_ids, ds.cat_ids, ds.zone_ids, ds.prices)
    build_sec = time.time() - t0
    print(f"[index] built IVF with 256 lists for {len(ds.vectors)} items in {build_sec:.2f}s")

    estimator = SelectivityEstimator(index)
    planner = CostBasedQueryPlanner(index, estimator)
    tau_low, tau_high = planner.calibrate_thresholds(ds.vectors[:50])
    print(f"[planner] calibrated thresholds: tau_low={tau_low:.3f}, tau_high={tau_high:.2f}")

    post_baseline = PurePostFilterBaseline(index, top_m=100, fixed_nprobe=16)
    pre_baseline = PurePreFilterBaseline(index)

    workload = build_bucket_workload(ds, index, n_queries=80)

    print("\n" + "=" * 90)
    header = f"{'Selectivity Bucket':<24} | {'System':<20} | {'Recall@10':<9} | {'p50 (ms)':<8} | {'p95 (ms)':<8} | {'Strategy'}"
    print(header)
    print("=" * 90)

    for b_name, queries in workload:
        # compute exact brute force ground truths first
        gts = [exact_ground_truth(index, q, f, k=10)[0] for q, f in queries]

        systems = [
            ("Pure Post-Filter", lambda q, f: (*post_baseline.search(q, f, k=10), "POST_FILTER")),
            ("Pure Pre-Filter", lambda q, f: (*pre_baseline.search(q, f, k=10), "EXACT_SCAN")),
            ("Our Hybrid Planner", lambda q, f: planner.search(q, f, k=10)[:3]),
        ]

        for s_name, runner in systems:
            recalls = []
            lats = []
            routes = {}

            for (q, f), gt in zip(queries, gts):
                t_start = time.perf_counter()
                pred_ids, _, route = runner(q, f)
                dur_ms = (time.perf_counter() - t_start) * 1000.0

                lats.append(dur_ms)
                recalls.append(compute_recall_at_k(pred_ids, gt, k=10))
                routes[route] = routes.get(route, 0) + 1

            avg_rec = float(np.mean(recalls))
            p50 = float(np.percentile(lats, 50))
            p95 = float(np.percentile(lats, 95))
            main_route = max(routes.items(), key=lambda x: x[1])[0]

            print(f"{b_name:<24} | {s_name:<20} | {avg_rec:>9.4f} | {p50:>8.2f} | {p95:>8.2f} | {main_route}")
        print("-" * 90)

    run_part6_planner_breaker(ds, index, estimator)


def run_part6_planner_breaker(ds: CatalogueDataset, index: CustomIVFIndex, estimator: SelectivityEstimator):
    print("\n" + "=" * 90)
    print(" Part 6: Planner Failure Mode (Correlated Predicate Trap & Micro-Sample Fix)")
    print("=" * 90)

    # Correlated trap: Mumbai + smartphones + budget price <= 6500
    # In independence assumption: P(Mumbai)*P(smartphones)*P(<=6500) gives ~1.8% (> tau_low=0.010)
    # Naive planner expects ~900 items and schedules an IVF partition scan.
    # But smartphones in Mumbai are heavily skewed to premium models; only ~3 budget items exist!
    trap_filter = FilterSpec(
        zone_id=ds.zone_to_id["Mumbai_Bhiwandi"],
        cat_id=ds.cat_to_id["smartphones"],
        max_price=6500.0,
    )

    true_matches = int(np.sum(index.global_filter_mask(trap_filter)))
    total_alive = int(np.sum(index.alive))
    true_sel = true_matches / float(total_alive)

    ind_est = estimator.estimate_independent(trap_filter)
    sketch_est = estimator.estimate_with_sketch(trap_filter)

    # test with planner threshold tau_low = 0.010
    naive_planner = CostBasedQueryPlanner(index, estimator, tau_low=0.010, use_sketch_fix=False)
    fixed_planner = CostBasedQueryPlanner(index, estimator, tau_low=0.010, use_sketch_fix=True)

    wrong_route, _, _ = naive_planner.choose_strategy(trap_filter)
    right_route, _, _ = fixed_planner.choose_strategy(trap_filter)

    print(f"Filter tested       : {trap_filter.describe(ds.categories, ds.zones)}")
    print(f"Independent Estimate: {ind_est:.5f} ({ind_est*100:.2f}%) -> Naive Route: {wrong_route} (Wasted IVF probe)")
    print(f"Actual Ground Truth : {true_sel:.5f} ({true_sel*100:.3f}% -> only {true_matches} matches)")
    print(f"Sketch-Aware Est    : {sketch_est:.5f} -> Fixed Route: {right_route} (Exact Pre-Filter)")
    print("=" * 90)


if __name__ == "__main__":
    run_benchmark()