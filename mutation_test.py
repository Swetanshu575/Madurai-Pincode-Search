# mutation_test.py
import time
import numpy as np
from data_prep import load_or_create_dataset
from ivf_index import CustomIVFIndex, FilterSpec
from baselines import exact_ground_truth
from planner import SelectivityEstimator, CostBasedQueryPlanner
from benchmark import compute_recall_at_k


def evaluate_slice(index: CustomIVFIndex, planner: CostBasedQueryPlanner, f: FilterSpec, n_queries=60, seed=77):
    rng = np.random.default_rng(seed)
    mask = index.global_filter_mask(f)
    valid_idx = np.flatnonzero(mask)
    recalls, lats = [], []

    for _ in range(n_queries):
        chosen = rng.choice(valid_idx)
        noise = rng.normal(0, 0.14, size=index.vectors.shape[1]).astype(np.float32)
        q = index.vectors[chosen] + noise
        q /= np.linalg.norm(q)

        gt_ids, _ = exact_ground_truth(index, q, f, k=10)
        t0 = time.perf_counter()
        pred_ids, _, _, _ = planner.search(q, f, k=10)
        lats.append((time.perf_counter() - t0) * 1000.0)
        recalls.append(compute_recall_at_k(pred_ids, gt_ids, k=10))

    return float(np.mean(recalls)), float(np.percentile(lats, 50)), float(np.percentile(lats, 95))


def run_mutation_experiment():
    print("=" * 85)
    print(" Part 5: Mutation Test (10% Mumbai Deletes + 5% Shifted Distribution Inserts)")
    print("=" * 85)

    ds = load_or_create_dataset(n_items=50_000)
    index = CustomIVFIndex(n_list=256, random_state=42)
    index.build(ds.vectors, ds.item_ids, ds.cat_ids, ds.zone_ids, ds.prices)

    estimator = SelectivityEstimator(index)
    planner = CostBasedQueryPlanner(index, estimator)
    planner.calibrate_thresholds(ds.vectors[:50])

    mumbai_id = ds.zone_to_id["Mumbai_Bhiwandi"]
    f_mumbai = FilterSpec(zone_id=mumbai_id)

    # 1. baseline measurements before mutation
    sizes_before = np.array([p.active_count for p in index.partitions])
    rec_pre, p50_pre, p95_pre = evaluate_slice(index, planner, f_mumbai)

    print(f"[Before] Active items   : {int(np.sum(index.alive)):,}")
    print(f"[Before] Partition size : mean={sizes_before.mean():.1f}, max={sizes_before.max()}, std={sizes_before.std():.1f}")
    print(f"[Before] Mumbai recall  : {rec_pre:.4f} | p50={p50_pre:.2f}ms | p95={p95_pre:.2f}ms")

    # 2. delete 10% of items in busiest zone (Mumbai_Bhiwandi)
    rng = np.random.default_rng(2026)
    mumbai_items = index.item_ids[(index.zone_ids == mumbai_id) & index.alive]
    n_delete = int(0.10 * len(mumbai_items))
    to_delete = rng.choice(mumbai_items, size=n_delete, replace=False)
    del_count = index.delete_items(to_delete)

    # 3. insert 5% new items drawn from shifted distribution (new category 20)
    n_insert = int(0.05 * len(ds.vectors))
    dim = ds.vectors.shape[1]
    shifted_anchor = rng.normal(0, 1.0, size=(1, dim)).astype(np.float32)
    shifted_anchor /= np.linalg.norm(shifted_anchor)

    new_vecs = shifted_anchor + rng.normal(0, 0.22, size=(n_insert, dim)).astype(np.float32)
    new_vecs /= np.linalg.norm(new_vecs, axis=1, keepdims=True)

    new_item_ids = np.arange(len(ds.vectors), len(ds.vectors) + n_insert, dtype=np.int32)
    new_cat_id = len(ds.categories)  # category 20
    new_cat_ids = np.full(n_insert, new_cat_id, dtype=np.int16)
    new_zone_ids = rng.choice(len(ds.zones), size=n_insert).astype(np.int16)
    new_prices = rng.uniform(3500.0, 14000.0, size=n_insert).astype(np.float32)

    assigned_cids = index.insert_items(new_vecs, new_item_ids, new_cat_ids, new_zone_ids, new_prices)
    estimator.refresh_stats(index)

    # 4. measurements after mutation
    sizes_after = np.array([p.active_count for p in index.partitions])
    unique_cids, cid_counts = np.unique(assigned_cids, return_counts=True)
    top_pileup = sorted(zip(unique_cids, cid_counts), key=lambda x: -x[1])[:3]

    rec_post, p50_post, p95_post = evaluate_slice(index, planner, f_mumbai)

    # test recall on the newly inserted shifted category
    f_shifted = FilterSpec(cat_id=new_cat_id, max_price=12000.0)
    rec_shifted, p50_sh, p95_sh = evaluate_slice(index, planner, f_shifted)

    print("\n" + "-" * 85)
    print(f"[Mutation] Deleted {del_count:,} Mumbai items; Inserted {n_insert:,} shifted items")
    print(f"[After]  Active items   : {int(np.sum(index.alive)):,}")
    print(f"[After]  Partition size : mean={sizes_after.mean():.1f}, max={sizes_after.max()}, std={sizes_after.std():.1f}")
    print(f"[Pile-Up] Top 3 partitions absorbed {sum(c for _, c in top_pileup):,} of {n_insert:,} new items: {[(int(c), int(cnt)) for c, cnt in top_pileup]}")
    print(f"[After]  Mumbai recall  : {rec_post:.4f} | p50={p50_post:.2f}ms | p95={p95_post:.2f}ms")
    print(f"[After]  Shifted recall : {rec_shifted:.4f} | p50={p50_sh:.2f}ms | p95={p95_sh:.2f}ms")
    print("=" * 85)


if __name__ == "__main__":
    run_mutation_experiment()