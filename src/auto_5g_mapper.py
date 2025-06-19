import os
import pandas as pd
from simulator import generate_users_within_bounds
from visualizer import plot_towers_on_map
from data_loader import load_opencellid_data

# مسار البيانات
DATA_DIR = "../data/"
OUTPUT_DIR = "../outputs/figures/"

# أخذ جميع الملفات .csv.gz داخل المجلد
all_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv.gz")]

for file in all_files:
    path = os.path.join(DATA_DIR, file)
    print(f"\n📂 تحليل الملف: {file}")

    # تحميل البيانات
    df = load_opencellid_data(path)

    # التحقق من وجود عمود radio
    if 'radio' not in df.columns:
        print("⚠️ العمود 'radio' غير موجود، نتجاهل الملف.")
        continue

    # فلترة 5G فقط
    df_5g = df[df['radio'].str.upper() == 'NR'].copy()

    if df_5g.empty:
        print("❌ لا توجد أبراج 5G (NR) في هذا الملف.")
        continue

    print(f"✅ عدد أبراج 5G: {len(df_5g)}")

    # حساب الحدود الجغرافية
    lat_min, lat_max = df_5g['lat'].min(), df_5g['lat'].max()
    lon_min, lon_max = df_5g['lon'].min(), df_5g['lon'].max()

    # توليد المستخدمين
    users = generate_users_within_bounds(lat_min, lat_max, lon_min, lon_max, count=1000)

    # تحديد مركز الخريطة
    center = [(lat_min + lat_max) / 2, (lon_min + lon_max) / 2]

    # اسم الخريطة الناتجة
    out_file = f"{OUTPUT_DIR}{file.replace('.csv.gz', '')}_5g_map.html"

    # رسم الخريطة
    plot_towers_on_map(df_5g, map_center=center, save_path=out_file, df_users=users)

    print(f"📍 تم حفظ الخريطة: {out_file}")
