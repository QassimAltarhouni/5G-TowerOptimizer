import numpy as np

def apply_knowledge_rules(individual, df_towers, df_users, distance_threshold=300):

    coords = df_towers[["lat", "lon"]].to_numpy()
    active_indices = np.where(individual == 1)[0]

    for i in range(len(active_indices)):
        for j in range(i + 1, len(active_indices)):
            idx1, idx2 = active_indices[i], active_indices[j]
            lat1, lon1 = coords[idx1]
            lat2, lon2 = coords[idx2]
            dist = np.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) * 111_000  # Approx meters

            if dist < distance_threshold:
                individual[idx2] = 0

    return individual
