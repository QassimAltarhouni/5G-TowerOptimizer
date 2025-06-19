import os
import pandas as pd
from simulator import generate_users_within_bounds
from visualizer import plot_towers_on_map
from data_loader import load_opencellid_data
from fitness_function import calculate_fitness

# إعدادات
DATA_DIR = "../data/"
OUTPUT_DIR = "../outputs/figures/"
CLEAN_DATA_DIR = "../outputs/clean_data/"

# تأكد أن المجلدات موجودة
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CLEAN_DATA_DIR, exist_ok=True)

# أسماء الملفات بعد إعادة التسمية اليدوية حسب الدولة
country_files = {
    "germany": "germany.csv.gz",
    "france": "france.csv.gz",
    "uk": "uk.csv.gz",
    "usa_310": "usa_310.csv.gz",
    "usa_311": "usa_311.csv.gz",
    "usa_312": "usa_312.csv.gz",
    "usa_313": "usa_313.csv.gz",
    "usa_314": "usa_314.csv.gz"
}

for country, filename in country_files.items():
    path = os.path.join(DATA_DIR, filename)
    print(f"\n📂 Processing: {country.upper()}")

    # تحميل البيانات
    df = load_opencellid_data(path)
    df.columns = df.columns.str.strip().str.lower()

    # فلترة أبراج 5G
    if 'radio' not in df.columns:
        print("⚠️ Column 'radio' missing, skipping...")
        continue

    df_5g = df[df['radio'].str.upper() == 'NR'].copy()
    if df_5g.empty:
        print("❌ No 5G towers found.")
        continue

    # حدود التغطية
    lat_min, lat_max = df_5g['lat'].min(), df_5g['lat'].max()
    lon_min, lon_max = df_5g['lon'].min(), df_5g['lon'].max()

    # تصنيف كثافة الأبراج
    tower_count = len(df_5g)
    if tower_count < 100:
        density = "low"
    elif tower_count < 500:
        density = "medium"
    else:
        density = "high"

    print(f"✅ Towers: {tower_count} | Density: {density.upper()}")

    # توليد المستخدمين
    users = generate_users_within_bounds(lat_min, lat_max, lon_min, lon_max, count=1000)

    # حفظ البيانات المفلترة
    df_5g.to_csv(os.path.join(CLEAN_DATA_DIR, f"{country}_5g_towers.csv"), index=False)
    users.to_csv(os.path.join(CLEAN_DATA_DIR, f"{country}_users.csv"), index=False)

    # حفظ الخريطة
    map_file = os.path.join(OUTPUT_DIR, f"{country}_5g_map.html")
    m = plot_towers_on_map(df_5g, map_center=[(lat_min + lat_max)/2, (lon_min + lon_max)/2], df_users=users)
    m.save(map_file)

    print(f"📍 Map saved: {map_file}")
    print(f"📊 Cleaned data saved to: {CLEAN_DATA_DIR}")

print("\n🎉 All instances processed successfully.")

results = calculate_fitness(df_towers=df_5g, df_users=users)

print(f"📈 Fitness for {country.upper()}:")
print(f"   Total Fitness Score: {results['fitness']}")
print(f"   Active Towers: {results['active_towers']}")
print(f"   Unserved Demand: {results['unserved_demand']} Mbps")
print(f"   Overload: {results['overload']} Mbps")
print(f"   Load Imbalance: {results['imbalance']} Mbps")