import numpy as np
import pandas as pd

def generate_users_within_bounds(lat_min, lat_max, lon_min, lon_max, count=1000):
    """
    Generate random users within a bounding box.
    Returns a DataFrame with lat, lon, and demand class.
    """
    lats = np.random.uniform(lat_min, lat_max, count)
    lons = np.random.uniform(lon_min, lon_max, count)
    user_classes = np.random.choice(['light', 'moderate', 'heavy'], size=count, p=[0.5, 0.3, 0.2])
    
    # Assign demand based on user class
    def demand_mbps(cls):
        if cls == 'light':
            return np.random.randint(1, 6)
        elif cls == 'moderate':
            return np.random.randint(6, 11)
        else:
            return np.random.randint(11, 31)

    demands = [demand_mbps(cls) for cls in user_classes]

    return pd.DataFrame({
        'lat': lats,
        'lon': lons,
        'user_class': user_classes,
        'demand_mbps': demands
    })
