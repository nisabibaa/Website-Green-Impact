import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import folium_static
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Judul Aplikasi
st.title("GreenImpact: Analisis Produktivitas Pertanian Indonesia")
st.markdown("---")

# Upload Data
st.sidebar.header("Upload Data Excel")
uploaded_file = st.sidebar.file_uploader("Pilih file Excel", type=["xlsx", "xls"])

if uploaded_file:
    # Membaca Data
    df = pd.read_excel(uploaded_file)
    st.subheader("Tampilan Data")
    st.write(df.head())
    
    # Menampilkan karakteristik variabel
    st.subheader("Statistik Deskriptif")
    st.write(df.describe())
    
    # Visualisasi Barchart untuk setiap variabel sebelum clustering
    st.subheader("Barchart Setiap Variabel")
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_columns = [col for col in numeric_columns if col not in ['Latitude', 'Longitude']]  # Hilangkan Latitude & Longitude
    for feature in numeric_columns:
        fig, ax = plt.subplots()
        sorted_df = df.sort_values(by=feature, ascending=False)
        sns.barplot(x=sorted_df['Provinsi'], y=sorted_df[feature], palette='viridis', ax=ax)
        ax.set_xlabel("Provinsi")
        ax.set_ylabel(feature)
        ax.set_title(f"Nilai {feature} per Provinsi")
        plt.xticks(rotation=90)
        st.pyplot(fig)
    
    # Memilih variabel untuk clustering
    st.sidebar.header("Konfigurasi Clustering")
    selected_features = st.sidebar.multiselect("Pilih Variabel untuk Clustering", numeric_columns, default=numeric_columns[:3])
    
    if selected_features:
        # Standarisasi Data
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df[selected_features])
        
        # Menentukan jumlah cluster
        k = st.sidebar.slider("Pilih Jumlah Cluster (K)", 2, 10, 3)
        
        # Clustering dengan K-Means
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        df['Cluster'] = kmeans.fit_predict(df_scaled) + 1  # Menjadikan cluster mulai dari 1
        
        # Menampilkan hasil clustering
        st.subheader("Hasil Clustering")
        st.write(df[['Cluster'] + selected_features])
        
        # Visualisasi Clustering
        st.subheader("Visualisasi Clustering")
        if len(selected_features) >= 2:
            fig, ax = plt.subplots()
            sns.scatterplot(x=df[selected_features[0]], y=df[selected_features[1]], hue=df['Cluster'], palette='viridis', ax=ax)
            ax.set_xlabel(selected_features[0])
            ax.set_ylabel(selected_features[1])
            st.pyplot(fig)
        else:
            st.warning("Silakan pilih setidaknya dua variabel numerik untuk visualisasi clustering.")
        
        # Interpretasi Hasil Clustering
        st.subheader("Hasil Clustering")
        cluster_means = df.groupby('Cluster')[selected_features].mean()
        st.write(cluster_means)
        st.markdown("Berikut adalah interpretasi dari hasil clustering:")
        cluster_order = cluster_means[selected_features[0]].sort_values().index.tolist()
        for i, cluster in enumerate(cluster_order, start=1):
            status = "rendah" if i == 1 else "tinggi" if i == len(cluster_order) else "sedang"
            st.markdown(f"- **Cluster {cluster}**: Produktivitas pertanian {status}.")
        
        # Visualisasi Peta
        st.subheader("Peta Hasil Clustering")
        try:
            df = df.dropna(subset=['Latitude', 'Longitude'])  # Hapus baris dengan koordinat kosong
            
            if not df.empty:
                m = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=5)
                colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightblue', 'lightgreen', 'gray', 'pink']
                for _, row in df.iterrows():
                    folium.CircleMarker(
                        location=[row['Latitude'], row['Longitude']],
                        radius=5,
                        color=colors[row['Cluster'] % len(colors)],
                        fill=True,
                        fill_color=colors[row['Cluster'] % len(colors)],
                        fill_opacity=0.7,
                        popup=f"Cluster: {row['Cluster']}"
                    ).add_to(m)
                folium_static(m)
            else:
                st.warning("Tidak ada data yang valid untuk ditampilkan di peta.")
        except Exception as e:
            st.error(f"Terjadi kesalahan dalam menampilkan peta: {e}")
    else:
        st.warning("Silakan pilih setidaknya satu variabel numerik untuk clustering.")
else:
    st.info("Silakan upload file Excel terlebih dahulu.")