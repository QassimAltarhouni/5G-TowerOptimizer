import numpy as np
from scipy.spatial import cKDTree

def apply_knowledge_rules(individual, df_towers, df_users, distance_threshold=300):
    """
    Apply spatial knowledge rules by disabling towers that are too close to each other.
    Uses a KD-tree for efficient pairwise distance checking.

    Parameters:
        individual (np.ndarray): Binary array indicating active towers.
        df_towers (pd.DataFrame): Tower data containing 'lat' and 'lon'.
        df_users (pd.DataFrame): (Unused here, but kept for compatibility).
        distance_threshold (float): Minimum allowed distance (in meters) between two active towers.

    Returns:
        np.ndarray: Modified individual after applying knowledge rules.
    """
    coords = df_towers[["lat", "lon"]].to_numpy()
    active_indices = np.where(individual == 1)[0]

    if len(active_indices) <= 1:
        return individual

    active_coords = coords[active_indices]

    # Build spatial tree and query pairs within distance
    tree = cKDTree(active_coords)
    # Convert meters to approximate degrees for geographic comparison
    approx_degree_threshold = distance_threshold / 111_000.0
    close_pairs = tree.query_pairs(r=approx_degree_threshold)

    for i, j in close_pairs:
        idx2 = active_indices[j]
        individual[idx2] = 0  # Deactivate one of the two close towers

    return individual
