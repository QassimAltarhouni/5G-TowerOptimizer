import numpy as np
from geopy.distance import geodesic

def calculate_fitness(df_towers, df_users, tower_capacity=1000, weights=(10, 5, 8, 1)):
    w1, w2, w3, w5 = weights

    # Step 1: assign each user to the closest tower within its range
    assignments = []
    for _, user in df_users.iterrows():
        best_tower = None
        best_dist = float('inf')

        for _, tower in df_towers.iterrows():
            dist = geodesic((user.lat, user.lon), (tower.lat, tower.lon)).meters
            if dist <= tower['range'] and dist < best_dist:
                best_tower = tower
                best_dist = dist

        if best_tower is not None:
            assignments.append((best_tower['cell'], user['demand_mbps']))
        else:
            assignments.append((None, user['demand_mbps']))

    # Step 2: compute total demand per tower
    load_by_tower = {}
    unserved_demand = 0

    for tower_id, demand in assignments:
        if tower_id is None:
            unserved_demand += demand
        else:
            load_by_tower[tower_id] = load_by_tower.get(tower_id, 0) + demand

    active_towers = len(load_by_tower)
    overload = sum(max(0, load - tower_capacity) for load in load_by_tower.values())
    imbalance = np.std(list(load_by_tower.values())) if load_by_tower else 0

    fitness = (w1 * active_towers) + (w2 * unserved_demand) + (w3 * overload) + (w5 * imbalance)

    return {
        "fitness": round(fitness, 2),
        "active_towers": active_towers,
        "unserved_demand": round(unserved_demand, 2),
        "overload": round(overload, 2),
        "imbalance": round(imbalance, 2)
    }
