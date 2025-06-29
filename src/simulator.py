import numpy as np
import pandas as pd

def generate_users_near_towers(df_towers, count=100000, max_distance_km=2.0):
    """
    Generate users within a fixed circular range of selected tower locations.

    Parameters:
        df_towers (DataFrame): Tower data with 'lat' and 'lon' columns.
        count (int): Total number of users to generate.
        max_distance_km (float): Max distance from tower (in km).

    Returns:
        DataFrame: User records with lat, lon, class, and demand.
    """
    users = []

    tower_sample = df_towers.sample(n=count, replace=True)

    for i in range(count):
        tower = tower_sample.iloc[i]
        # Random angle and radius
        angle = np.random.uniform(0, 2 * np.pi)
        radius_km = np.random.uniform(0, max_distance_km)

        # Convert radius to degrees
        delta_lat = (radius_km / 111) * np.cos(angle)
        delta_lon = (radius_km / (111 * np.cos(np.radians(tower['lat'])))) * np.sin(angle)

        lat = tower['lat'] + delta_lat
        lon = tower['lon'] + delta_lon

        user_class = np.random.choice(['light', 'moderate', 'heavy'], p=[0.5, 0.3, 0.2])
        demand = {
            'light': np.random.randint(1, 6),
            'moderate': np.random.randint(6, 11),
            'heavy': np.random.randint(11, 31)
        }[user_class]

        users.append({
            'lat': lat,
            'lon': lon,
            'user_class': user_class,
            'demand_mbps': demand
        })

    return pd.DataFrame(users)
