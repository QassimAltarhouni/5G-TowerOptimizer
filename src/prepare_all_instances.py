# prepare_all_instances.py

import os
import pandas as pd
import numpy as np
from simulator import generate_users_near_towers
from visualizer import plot_towers_on_map
from data_loader import load_opencellid_data
from fitness_function import calculate_fitness, compute_normalization_bounds
from real_user_generator import generate_users_from_population_raster
from genetic_optimizer import run_ga
from genetic_optimizer import run_kbga



# === SETTINGS ===
DATA_DIR = "../data/"
OUTPUT_DIR = "../outputs/figures/"
CLEAN_DATA_DIR = "../outputs/clean_data/"
COUNTRY = "Germany"
FILENAME = "germany.csv.gz"
TIF_PATH = "../data/deu_ppp_2020_constrained.tif"

# === CREATE DIRECTORIES ===
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CLEAN_DATA_DIR, exist_ok=True)

# === LOAD TOWER DATA ===
print(f"\n📂 Processing: {COUNTRY.upper()}")
path = os.path.join(DATA_DIR, FILENAME)
df = load_opencellid_data(path)
df.columns = df.columns.str.strip().str.lower()

if 'radio' not in df.columns:
    print("⚠️ Column 'radio' missing, aborting...")
    exit()

# === FILTER FOR 5G TOWERS ===
df_5g = df[df['radio'].str.upper() == 'NR'].copy()
if df_5g.empty:
    print("❌ No 5G towers found.")
    exit()

# === TOWER STATISTICS ===
lat_min, lat_max = df_5g['lat'].min(), df_5g['lat'].max()
lon_min, lon_max = df_5g['lon'].min(), df_5g['lon'].max()

tower_count = len(df_5g)
density = "low" if tower_count < 100 else "medium" if tower_count < 500 else "high"
print(f"✅ Towers: {tower_count} | Density: {density.upper()}")

# === GENERATE OR LOAD USER DEMAND ===
user_csv_path = os.path.join(CLEAN_DATA_DIR, f"{COUNTRY}_users.csv")

if os.path.exists(user_csv_path):
    print("📂 Loading existing users from file...")
    users = pd.read_csv(user_csv_path)
else:
    print("🧬 Generating new user population...")
    users = generate_users_from_population_raster(TIF_PATH, sample_count=100000)
    users.to_csv(user_csv_path, index=False)
    print(f"✅ Saved users to {user_csv_path}")

# === SAVE CLEANED DATA ===
df_5g.to_csv(os.path.join(CLEAN_DATA_DIR, f"{COUNTRY}_5g_towers.csv"), index=False)
users.to_csv(os.path.join(CLEAN_DATA_DIR, f"{COUNTRY}_users.csv"), index=False)

# === PLOT MAP ===
map_file = os.path.join(OUTPUT_DIR, f"{COUNTRY}_5g_map.html")
m = plot_towers_on_map(df_5g, map_center=[(lat_min + lat_max) / 2, (lon_min + lon_max) / 2], df_users=users)
m.save(map_file)
print(f"📍 Map saved: {map_file}")
print(f"📊 Cleaned data saved to: {CLEAN_DATA_DIR}")

# === DYNAMIC NORMALIZATION BOUNDS ===
print("\n📊 Sampling fitness for normalization bounds...")
normalization_bounds = compute_normalization_bounds(df_5g, users)
min_vals, max_vals = normalization_bounds

print("\n📏 Normalization bounds:")
print("   Min:", min_vals)
print("   Max:", max_vals)

# === BASELINE FITNESS CALCULATION ===
print("\n⚙️ Calculating baseline fitness...")
results = calculate_fitness(
    df_towers=df_5g.copy(),
    df_users=users,
    normalization_bounds=normalization_bounds,
    verbose=True,
)

print(f"\n📈 Fitness for {COUNTRY.upper()}:")
print(f"   Total Fitness Score: {results['fitness']}")
print(f"   Active Towers: {results['active_towers']}")
print(f"   Unserved Demand: {results['unserved_demand']} Mbps")
print(f"   Overload: {results['overload']} Mbps")
print(f"   Excessive Distance: {results['excessive_distance']} meters")
print(f"   Load Imbalance: {results['imbalance']} Mbps")

# === GENETIC OPTIMIZATION ===
print("\n🚀 Running Genetic Algorithm optimization...")
best_solution, best_score = run_ga(
    df_5g,
    users,
    normalization_bounds=normalization_bounds,
)

print("\n🏁 FINAL BEST FITNESS:", best_score)

# === KNOWLEDGE-BASED GENETIC OPTIMIZATION ===
kbga_solution, kbga_score = run_kbga(
    df_5g,
    users,
    normalization_bounds=normalization_bounds,
)

print("\n🧠 FINAL KBGA FITNESS:", kbga_score)

# === COMPARISON ===
print("\n📊 COMPARISON:")
print(f"   GA Score   : {best_score}")
print(f"   KBGA Score : {kbga_score}")

comparison_path = f"../outputs/{COUNTRY}_comparison.csv"
pd.DataFrame([{
    "Method": "GA",
    "Fitness": best_score
}, {
    "Method": "KBGA",
    "Fitness": kbga_score
}]).to_csv(comparison_path, index=False)
print(f"📁 Comparison saved to: {comparison_path}")
