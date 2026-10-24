# data_prep.py
import os
import time
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional

# 20 product categories
CATEGORIES = [
    "cotton_saree", "silk_saree", "banarasi_saree", "kurta_set", "lehenga_choli",
    "mens_sherwani", "ethnic_footwear", "brass_pooja_thali", "diwali_led_lights", "terracotta_diya",
    "dry_fruits_gift_box", "sweets_mithai", "smartphones", "tws_earbuds", "smart_watches",
    "pressure_cooker", "mixer_grinder", "bedsheets_cotton", "gold_plated_jewellery", "handloom_dupatta"
]

# price log-normal params (median INR, sigma) per category
CATEGORY_PRICE_PARAMS = {
    "cotton_saree": (1100.0, 0.45),
    "silk_saree": (4800.0, 0.55),
    "banarasi_saree": (6200.0, 0.50),
    "kurta_set": (1400.0, 0.50),
    "lehenga_choli": (5500.0, 0.60),
    "mens_sherwani": (4500.0, 0.55),
    "ethnic_footwear": (850.0, 0.45),
    "brass_pooja_thali": (650.0, 0.40),
    "diwali_led_lights": (280.0, 0.45),
    "terracotta_diya": (180.0, 0.35),
    "dry_fruits_gift_box": (1250.0, 0.45),
    "sweets_mithai": (550.0, 0.35),
    "smartphones": (16500.0, 0.50),
    "tws_earbuds": (1800.0, 0.55),
    "smart_watches": (2400.0, 0.50),
    "pressure_cooker": (1900.0, 0.40),
    "mixer_grinder": (3200.0, 0.45),
    "bedsheets_cotton": (950.0, 0.40),
    "gold_plated_jewellery": (1500.0, 0.60),
    "handloom_dupatta": (750.0, 0.40),
}

# 30 delivery zones with zipf-like skew
# metro hubs hold ~68% of stock, small tier-3 zones have <0.1% to 0.2%
ZONES_WITH_WEIGHTS = [
    ("Mumbai_Bhiwandi", 0.2600),
    ("Delhi_Gurugram", 0.1800),
    ("Bengaluru_Hosur", 0.1400),
    ("Kolkata_Hub", 0.1000),
    ("Hyderabad_Hub", 0.0650),
    ("Chennai_Outer", 0.0450),
    ("Pune_Chakan", 0.0350),
    ("Ahmedabad_Sanand", 0.0300),
    ("Surat_Textile", 0.0250),
    ("Jaipur_Hub", 0.0200),
    ("Lucknow_Hub", 0.0180),
    ("Kanpur_Zone", 0.0120),
    ("Nagpur_Central", 0.0110),
    ("Indore_Hub", 0.0100),
    ("Coimbatore_TN", 0.0080),
    ("Patna_Zone", 0.0060),
    ("Bhopal_Zone", 0.0050),
    ("Ludhiana_Hub", 0.0040),
    ("Guwahati_NE", 0.0035),
    ("Visakhapatnam_AP", 0.0030),
    ("Varanasi_UP", 0.0028),
    ("Madurai_TN", 0.0022),     # ~110 items in 50k
    ("Mysuru_KA", 0.0020),
    ("Ranchi_JH", 0.0018),
    ("Jodhpur_RJ", 0.0015),
    ("Raipur_CG", 0.0012),
    ("Siliguri_WB", 0.0009),
    ("Tinsukia_AS", 0.0008),
    ("PortBlair_AN", 0.0007),
    ("Kargil_LK", 0.0006),
]


@dataclass
class CatalogueDataset:
    vectors: np.ndarray
    cat_ids: np.ndarray
    zone_ids: np.ndarray
    prices: np.ndarray
    item_ids: np.ndarray
    categories: List[str]
    zones: List[str]
    cat_to_id: Dict[str, int]
    zone_to_id: Dict[str, int]
    subcluster_centers: np.ndarray


def generate_bharat_bazaar_dataset(
    n_items: int = 50_000,
    dim: int = 128,
    seed: int = 42,
    save_path: Optional[str] = "catalogue_data.npz",
) -> CatalogueDataset:
    t0 = time.time()
    rng = np.random.default_rng(seed)

    zones = [z[0] for z in ZONES_WITH_WEIGHTS]
    zone_weights = np.array([z[1] for z in ZONES_WITH_WEIGHTS], dtype=np.float64)
    zone_weights /= zone_weights.sum()

    cat_to_id = {c: i for i, c in enumerate(CATEGORIES)}
    zone_to_id = {z: i for i, z in enumerate(zones)}

    num_cats = len(CATEGORIES)
    num_zones = len(zones)
    subclusters_per_cat = 5

    # generate category anchor vectors & 5 sub-clusters each
    cat_anchors = rng.normal(0, 1.0, size=(num_cats, dim)).astype(np.float32)
    cat_anchors /= np.linalg.norm(cat_anchors, axis=1, keepdims=True)

    subcluster_centers = np.zeros((num_cats, subclusters_per_cat, dim), dtype=np.float32)
    for c in range(num_cats):
        offsets = rng.normal(0, 0.28, size=(subclusters_per_cat, dim)).astype(np.float32)
        centers = cat_anchors[c] + offsets
        centers /= np.linalg.norm(centers, axis=1, keepdims=True)
        subcluster_centers[c] = centers

    # sample zones with power-law skew
    zone_ids = rng.choice(num_zones, size=n_items, p=zone_weights).astype(np.int16)

    # base category probabilities
    base_cat_probs = np.array([
        0.08, 0.05, 0.04, 0.09, 0.04,
        0.03, 0.06, 0.05, 0.07, 0.05,
        0.05, 0.05, 0.06, 0.06, 0.05,
        0.04, 0.03, 0.04, 0.03, 0.03
    ], dtype=np.float64)
    base_cat_probs /= base_cat_probs.sum()

    # conditional category distribution per zone (injecting regional skew)
    zone_cat_probs = np.tile(base_cat_probs, (num_zones, 1))

    # tamil nadu zones: heavy skew toward cotton and silk sarees
    for tn_zone in ["Madurai_TN", "Coimbatore_TN", "Chennai_Outer"]:
        zid = zone_to_id[tn_zone]
        zone_cat_probs[zid] *= 0.25
        zone_cat_probs[zid, cat_to_id["cotton_saree"]] = 0.48
        zone_cat_probs[zid, cat_to_id["silk_saree"]] = 0.24
        zone_cat_probs[zid] /= zone_cat_probs[zid].sum()

    # textile belts
    for tz in ["Surat_Textile", "Varanasi_UP"]:
        zid = zone_to_id[tz]
        zone_cat_probs[zid] *= 0.30
        zone_cat_probs[zid, cat_to_id["banarasi_saree"]] = 0.35
        zone_cat_probs[zid, cat_to_id["cotton_saree"]] = 0.25
        zone_cat_probs[zid, cat_to_id["handloom_dupatta"]] = 0.15
        zone_cat_probs[zid] /= zone_cat_probs[zid].sum()

    cat_ids = np.zeros(n_items, dtype=np.int16)
    for zid in range(num_zones):
        mask = (zone_ids == zid)
        cnt = int(mask.sum())
        if cnt > 0:
            cat_ids[mask] = rng.choice(num_cats, size=cnt, p=zone_cat_probs[zid])

    # build item vectors around category subclusters + small zone drift
    zone_drift = rng.normal(0, 0.06, size=(num_zones, dim)).astype(np.float32)
    sub_ids = rng.integers(0, subclusters_per_cat, size=n_items)

    base_vectors = subcluster_centers[cat_ids, sub_ids]
    noise = rng.normal(0, 0.16, size=(n_items, dim)).astype(np.float32)
    vectors = base_vectors + zone_drift[zone_ids] + noise

    # l2 normalize for cosine similarity
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = (vectors / np.maximum(norms, 1e-12)).astype(np.float32)

    # generate prices correlated with category
    prices = np.zeros(n_items, dtype=np.float32)
    for cid, cname in enumerate(CATEGORIES):
        med, sigma = CATEGORY_PRICE_PARAMS[cname]
        c_mask = (cat_ids == cid)
        c_count = int(c_mask.sum())
        if c_count > 0:
            raw = rng.lognormal(mean=np.log(med), sigma=sigma, size=c_count)
            prices[c_mask] = np.clip(raw, 99.0, 95_000.0)

    # subtle jaipur lehenga trap for correlation check
    jl_mask = (zone_ids == zone_to_id["Jaipur_Hub"]) & (cat_ids == cat_to_id["lehenga_choli"])
    if jl_mask.sum() > 0:
        idxs = np.flatnonzero(jl_mask)
        prices[idxs] = rng.uniform(6500.0, 22000.0, size=len(idxs)).astype(np.float32)
        if len(idxs) >= 4:
            prices[idxs[:3]] = np.array([1850.0, 2100.0, 2399.0], dtype=np.float32)

    prices = np.round(prices, 2)
    item_ids = np.arange(n_items, dtype=np.int32)

    if save_path:
        np.savez_compressed(
            save_path,
            vectors=vectors,
            cat_ids=cat_ids,
            zone_ids=zone_ids,
            prices=prices,
            item_ids=item_ids,
            categories=np.array(CATEGORIES),
            zones=np.array(zones),
            subcluster_centers=subcluster_centers,
        )

    elapsed = time.time() - t0
    print(f"[data] generated {n_items} items in {elapsed:.2f}s")

    return CatalogueDataset(
        vectors=vectors,
        cat_ids=cat_ids,
        zone_ids=zone_ids,
        prices=prices,
        item_ids=item_ids,
        categories=CATEGORIES,
        zones=zones,
        cat_to_id=cat_to_id,
        zone_to_id=zone_to_id,
        subcluster_centers=subcluster_centers,
    )


def load_or_create_dataset(path: str = "catalogue_data.npz", n_items: int = 50_000) -> CatalogueDataset:
    if os.path.exists(path):
        data = np.load(path, allow_pickle=False)
        categories = data["categories"].tolist()
        zones = data["zones"].tolist()
        return CatalogueDataset(
            vectors=data["vectors"],
            cat_ids=data["cat_ids"],
            zone_ids=data["zone_ids"],
            prices=data["prices"],
            item_ids=data["item_ids"],
            categories=categories,
            zones=zones,
            cat_to_id={c: i for i, c in enumerate(categories)},
            zone_to_id={z: i for i, z in enumerate(zones)},
            subcluster_centers=data["subcluster_centers"],
        )
    return generate_bharat_bazaar_dataset(n_items=n_items, save_path=path)


if __name__ == "__main__":
    ds = generate_bharat_bazaar_dataset()
    m_id = ds.zone_to_id["Madurai_TN"]
    s_id = ds.cat_to_id["cotton_saree"]
    m_count = np.sum((ds.zone_ids == m_id) & (ds.cat_ids == s_id) & (ds.prices <= 1500))
    print(f"Sample check: Madurai cotton sarees under Rs. 1500 = {m_count} items")