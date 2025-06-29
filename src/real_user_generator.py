import rasterio
import numpy as np
import pandas as pd
from rasterio.transform import xy

def generate_users_from_population_raster(tif_path, sample_count=100000):
    with rasterio.open(tif_path) as src:
        data = src.read(1)
        transform = src.transform

        rows, cols = np.where(data > 0)
        populations = data[rows, cols]

        latitudes = []
        longitudes = []
        for r, c in zip(rows, cols):
            lon, lat = xy(transform, r, c)
            latitudes.append(lat)
            longitudes.append(lon)

        df = pd.DataFrame({
            "lat": latitudes,
            "lon": longitudes,
            "population": populations
        })

        weights = df["population"] / df["population"].sum()

        user_points = df.sample(n=sample_count, weights=weights, replace=True)

        user_classes = np.random.choice(
            ['light', 'moderate', 'heavy'],
            size=sample_count,
            p=[0.5, 0.3, 0.2]
        )

        demand = []
        for c in user_classes:
            if c == "light":
                demand.append(np.random.randint(1, 6))
            elif c == "moderate":
                demand.append(np.random.randint(6, 11))
            else:
                demand.append(np.random.randint(11, 31))

        user_points = user_points.reset_index(drop=True)
        user_points["user_class"] = user_classes
        user_points["demand_mbps"] = demand

        return user_points
