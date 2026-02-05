import streamlit as st
import pandas as pd

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 1. Veri Yükleme (Önceki dosyalardan okuyoruz)
@st.cache_data
def load_data():
    movies = pd.read_csv('movies.csv')
    ratings = pd.read_csv('ratings.csv')
    return pd.merge(ratings, movies, on='movieId'), movies

df, movies_df = load_data()

# Hız için pivot tabloyu oluşturuyoruz (User-Movie Matrix)
# Satırlar: Kullanıcılar, Sütunlar: Film İsimleri
user_movie_matrix = df.pivot_table(index='userId', columns='title', values='rating').fillna(0)

st.title("🎬 Zevk İkizi ve Fark Analizi")

# 2. Kullanıcı Seçimi
selected_user = st.sidebar.selectbox("Kendi Kullanıcı ID'nizi Seçin:", user_movie_matrix.index)

if selected_user:
    # Kullanıcı vektörünü al
    user_vec = user_movie_matrix.loc[selected_user].values.reshape(1, -1)
    
    # Tüm kullanıcılarla benzerliği (açıyı) hesapla
    similarities = cosine_similarity(user_vec, user_movie_matrix.values).flatten()
    sim_series = pd.Series(similarities, index=user_movie_matrix.index).drop(selected_user)
    
    # En yakın "Zevk İkizini" bul
    best_match_id = sim_series.idxmax()
    similarity_score = sim_series.max()

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👤 Senin Profilin")
        my_top_films = user_movie_matrix.loc[selected_user].sort_values(ascending=False).head(5)
        st.write("En çok puan verdiğin filmler:")
        st.write(", ".join(my_top_films[my_top_films > 0].index.tolist()))

    with col2:
        st.subheader("👯 Zevk İkizin")
        st.write(f"**Kullanıcı {best_match_id}** ile zevkleriniz **%{similarity_score*100:.1f}** oranında aynı doğrultuda.")
        match_top_films = user_movie_matrix.loc[best_match_id].sort_values(ascending=False).head(5)
        st.write("Onun en sevdiği filmler:")
        st.write(", ".join(match_top_films[match_top_films > 0].index.tolist()))

    st.divider()

    # 3. SENİN MANTIĞIN: Ortakları Çıkar, Farklı Olanı Öner
    st.subheader(f"✨ Kullanıcı {best_match_id}'den Sana Özel Öneriler")
    
    # Kural 1: Benim izlemediğim (puanım 0 olan)
    # Kural 2: Onun çok sevdiği (puanı 4 veya 5 olan)
    my_ratings = user_movie_matrix.loc[selected_user]
    match_ratings = user_movie_matrix.loc[best_match_id]
    
    # Ortak izlediklerimizi filtrele ve sadece onun bildiği "farklı" filmleri al
    recommendations = match_ratings[(my_ratings == 0) & (match_ratings >= 4)].sort_values(ascending=False)

    if not recommendations.empty:
        st.write("Senin henüz keşfetmediğin ama ikizinin bayıldığı şu filmler tam sana göre:")
        
        # Daha havalı bir sunum için sütunlara bölelim
        rec_list = recommendations.index.tolist()[:6]
        cols = st.columns(3)
        for i, movie in enumerate(rec_list):
            cols[i % 3].info(f"🎞️ {movie}")
            
        # Neden Öneriyoruz Açıklaması
        st.caption(f"💡 Not: Bu filmler, Kullanıcı {best_match_id} ile aranızdaki **'bilgi farkından'** süzülerek gelmiştir.")
    else:
        st.warning("İnanılmaz! Zevk ikizinin izlediği her şeyi sen de izlemişsin. Yeni bir ikiz aramalıyız.")

    # 4. Doğrultu Kanıtı (Bonus Görselleştirme)
    st.divider()
    st.subheader("📊 Ortak Nokta Analizi")
    common_movies = user_movie_matrix.loc[[selected_user, best_match_id], (user_movie_matrix.loc[selected_user] > 0) & (user_movie_matrix.loc[best_match_id] > 0)]
    if not common_movies.empty:
        st.write("İkinizin de izleyip benzer puanlar verdiği filmler (Bu sizin 'Aynı Doğrultu'da olduğunuzun ispatıdır):")
        st.dataframe(common_movies.T.head(10))