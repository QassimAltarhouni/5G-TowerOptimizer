import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def latlon_to_xyz(lat, lon):
    """Convert latitude and longitude arrays to 3D Cartesian coordinates."""
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    return np.vstack((x, y, z)).T

def calculate_fitness(
    df_towers,
    df_users,
    tower_capacity=1000,
    weights=(10, 5, 8, 0.5, 1),
    verbose=True,
    normalization_bounds=None
):
    w1, w2, w3, w4, w5 = weights

    df_towers['range'] = pd.to_numeric(df_towers.get('range', 2000), errors='coerce').fillna(2000)
    df_towers = df_towers[df_towers['range'] > 0]

    assignments = []
    excessive_distance_penalty = 0
    unserved_demand = 0

    # === Fast nearest tower search using a KD-tree ===
    tower_xyz = latlon_to_xyz(df_towers["lat"].to_numpy(), df_towers["lon"].to_numpy())
    user_xyz = latlon_to_xyz(df_users["lat"].to_numpy(), df_users["lon"].to_numpy())

    tree = cKDTree(tower_xyz)
    chord_dist, closest_indices = tree.query(user_xyz, k=1)
    min_distances = 2 * np.arcsin(np.clip(chord_dist / 2, 0, 1)) * 6371000
    closest_cells = df_towers.iloc[closest_indices]["cell"].to_numpy()

    demands = df_users["demand_mbps"].to_numpy()

    served_mask = min_distances <= 5000

    penalties = np.zeros_like(min_distances)
    mask_2000 = (min_distances > 2000) & (min_distances <= 3000)
    penalties[mask_2000] = (min_distances[mask_2000] - 2000) * 0.5
    mask_3000 = (min_distances > 3000) & (min_distances <= 5000)
    penalties[mask_3000] = (min_distances[mask_3000] - 2000) * 1.5

    excessive_distance_penalty = penalties[served_mask].sum()
    unserved_demand = demands[~served_mask].sum()

    assignments = list(
        zip(closest_cells[served_mask], demands[served_mask])
    )

    load_by_tower = {}
    for tower_id, demand in assignments:
        if tower_id is None:
            unserved_demand += demand
        else:
            load_by_tower[tower_id] = load_by_tower.get(tower_id, 0) + demand

    active_towers = len(load_by_tower)
    overload = sum(max(0, load - tower_capacity) for load in load_by_tower.values())
    imbalance = np.std(list(load_by_tower.values())) if load_by_tower else 0

    if normalization_bounds:
        min_vals, max_vals = normalization_bounds

        def normalize(val, min_val, max_val):
            if max_val == min_val:
                return 0.0
            norm_val = (val - min_val) / (max_val - min_val)
            return float(np.clip(norm_val, 0.0, 1.0))

        norm_active = normalize(
            active_towers, min_vals["active_towers"], max_vals["active_towers"]
        )
        norm_unserved = normalize(
            unserved_demand, min_vals["unserved_demand"], max_vals["unserved_demand"]
        )
        norm_overload = normalize(overload, min_vals["overload"], max_vals["overload"])
        norm_distance = normalize(
            excessive_distance_penalty,
            min_vals["excessive_distance"],
            max_vals["excessive_distance"],
        )
        norm_imbalance = normalize(imbalance, min_vals["imbalance"], max_vals["imbalance"])

        fitness = (
                (w1 * norm_active)
                + (w2 * norm_unserved)
                + (w3 * norm_overload)
                + (w4 * norm_distance)
                + (w5 * norm_imbalance)
        )
    else:
        fitness = (
                (w1 * active_towers)
                + (w2 * unserved_demand)
                + (w3 * overload)
                + (w4 * excessive_distance_penalty)
                + (w5 * imbalance)
        )

    if verbose:
        print(" Breakdown:")
        if normalization_bounds:
            print(
                f"   Active Towers: {active_towers} × {w1} = {w1 * active_towers} ({norm_active:.4f})"
            )
            print(
                f"   Unserved Demand: {unserved_demand} Mbps × {w2} = {w2 * unserved_demand} ({norm_unserved:.4f})"
            )
            print(
                f"   Overload: {overload} Mbps × {w3} = {w3 * overload} ({norm_overload:.4f})"
            )
            print(
                f"   Excessive Distance: {excessive_distance_penalty:.2f} m × {w4} = {w4 * excessive_distance_penalty:.2f} ({norm_distance:.4f})"
            )
            print(
                f"   Load Imbalance: {imbalance:.2f} Mbps × {w5} = {w5 * imbalance:.2f} ({norm_imbalance:.4f})"
            )
        else:
            print(
                f"   Active Towers: {active_towers} × {w1} = {w1 * active_towers}"
            )
            print(
                f"   Unserved Demand: {unserved_demand} Mbps × {w2} = {w2 * unserved_demand}"
            )
            print(f"   Overload: {overload} Mbps × {w3} = {w3 * overload}")
            print(
                f"   Excessive Distance: {excessive_distance_penalty:.2f} m × {w4} = {w4 * excessive_distance_penalty:.2f}"
            )
            print(
                f"   Load Imbalance: {imbalance:.2f} Mbps × {w5} = {w5 * imbalance:.2f}"
            )

    return {
        "fitness": round(fitness, 6),
        "active_towers": active_towers,
        "unserved_demand": round(unserved_demand, 2),
        "overload": round(overload, 2),
        "excessive_distance": round(excessive_distance_penalty, 2),
        "imbalance": round(imbalance, 2)
    }


def compute_normalization_bounds(df_towers, df_users, samples=5, frac_range=(0.3, 0.9)):
    """Estimate normalization bounds using random tower samples."""

    metrics = [
        "active_towers",
        "unserved_demand",
        "overload",
        "excessive_distance",
        "imbalance",
    ]

    results = []
    for _ in range(samples):
        frac = np.random.uniform(frac_range[0], frac_range[1])
        sampled = df_towers.sample(frac=frac)
        stats = calculate_fitness(sampled, df_users, verbose=False)
        results.append(stats)

    # include full dataset to prevent values outside sampled range
    results.append(calculate_fitness(df_towers, df_users, verbose=False))

    min_vals = {m: min(0, *(r[m] for r in results)) for m in metrics}
    max_vals = {m: max(r[m] for r in results) for m in metrics}

    return min_vals, max_vals