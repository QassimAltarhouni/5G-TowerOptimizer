import folium

def plot_towers_on_map(df_towers, map_center=[52.52, 13.405], zoom=11, save_path=None, df_users=None):
    m = folium.Map(location=map_center, zoom_start=zoom)

    # رسم الأبراج
    for _, row in df_towers.iterrows():
        popup_text = f"""
        <b>Radio:</b> {row.get('radio', '')}<br>
        <b>Range:</b> {row.get('range', 'N/A')} m<br>
        <b>Samples:</b> {row.get('samples', 'N/A')}<br>
        <b>Signal:</b> {row.get('averageSignal', 'N/A')}
        """
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=5,
            popup=folium.Popup(popup_text, max_width=300),
            color='blue',
            fill=True,
            fill_opacity=0.6
        ).add_to(m)

    # رسم المستخدمين حسب الفئة
    if df_users is not None:
        color_map = {'light': 'green', 'moderate': 'orange', 'heavy': 'red'}

        for _, row in df_users.iterrows():
            popup_text = f"""
            <b>User Class:</b> {row['user_class']}<br>
            <b>Demand:</b> {row['demand_mbps']} Mbps
            """
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=3,
                popup=folium.Popup(popup_text, max_width=250),
                color=color_map.get(row['user_class'], 'gray'),
                fill=True,
                fill_opacity=0.4
            ).add_to(m)

    if save_path:
        m.save(save_path)

    return m
