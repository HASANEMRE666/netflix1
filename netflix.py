import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 🧠 1. VERİ VE HAZIRLIK (DATA ENGINE)
# ==========================================

@st.cache_data
def load_and_process_data():
    # Dosya yollarını kendi dosyalarınla eşleştir
    movies = pd.read_csv('movies.csv')
    ratings = pd.read_csv('ratings.csv')
    
    # Başarı Oranı (Success Rate) Hesaplama: (Ortalama Puan / 5) * 100
    movie_stats = ratings.groupby('movieId')['rating'].agg(['mean', 'count']).reset_index()
    movie_stats['success_rate'] = (movie_stats['mean'] / 5 * 100).round(1)
    
    # Verileri birleştir
    movies = movies.merge(movie_stats, on='movieId', how='left').fillna(0)
    return movies, ratings

# ==========================================
# 🛠️ 2. ALGORİTMA MOTORLARI (RECOM ENGINES)
# ==========================================

# --- USER-BASED (Kullanıcı Benzerliği - Cosine Similarity) ---
def get_user_based(user_id, ratings_df, movies_df):
    # Performans için sadece popüler filmleri matrise al
    pop_movies = ratings_df.groupby('movieId').size()[lambda x: x > 30].index
    matrix = ratings_df[ratings_df['movieId'].isin(pop_movies)].pivot_table(
        index='userId', columns='movieId', values='rating').fillna(0)
    
    # Cosine Similarity ile ruh ikizini bul
    sim = cosine_similarity(matrix)
    sim_df = pd.DataFrame(sim, index=matrix.index, columns=matrix.index)
    soulmate_id = sim_df[user_id].sort_values(ascending=False).index[1]
    sim_score = sim_df.loc[user_id, soulmate_id]
    
    # Ruh ikizinin sevdiği ama senin izlemediğin filmler
    watched = ratings_df[ratings_df['userId'] == user_id]['movieId'].tolist()
    recoms = ratings_df[(ratings_df['userId'] == soulmate_id) & (ratings_df['rating'] >= 4.5)]
    final = recoms[~recoms['movieId'].isin(watched)].merge(movies_df, on='movieId')
    
    return final.sort_values('success_rate', ascending=False).head(5), soulmate_id, sim_score

# --- ITEM-BASED (Ürün Benzerliği - Davranışsal Korelasyon) ---
def get_item_based(user_id, ratings_df, movies_df):
    # Senin 5 yıldızlıların (Yoksa 4'ler)
    user_ratings = ratings_df[ratings_df['userId'] == user_id]
    top_films = user_ratings[user_ratings['rating'] == 5]
    if len(top_films) < 3:
        top_films = user_ratings[user_ratings['rating'] >= 4]
    
    # En güçlü 5 sinyal (Film) üzerinden benzerlerini bul
    seed_ids = top_films.sort_values('rating', ascending=False).head(5)['movieId'].tolist()
    
    recommendations = []
    for m_id in seed_ids:
        # Bu filmi seven (4.5+) diğer kullanıcılar
        fans = ratings_df[(ratings_df['movieId'] == m_id) & (ratings_df['rating'] >= 4.5)]['userId'].unique()
        # Bu fanların ortak sevdiği diğer filmler
        others = ratings_df[(ratings_df['userId'].isin(fans)) & (ratings_df['movieId'] != m_id)]
        recommendations.append(others)
    
    if not recommendations: return pd.DataFrame()
    
    combined = pd.concat(recommendations).groupby('movieId').size().reset_index(name='match_count')
    watched = user_ratings['movieId'].tolist()
    final = combined[~combined['movieId'].isin(watched)].merge(movies_df, on='movieId')
    
    return final.sort_values(['match_count', 'success_rate'], ascending=False).head(5)

# --- CONTENT-BASED (Tür Analizi - Metadata) ---
def get_content_based(user_id, ratings_df, movies_df):
    user_full = ratings_df[ratings_df['userId'] == user_id].merge(movies_df, on='movieId')
    # En çok izlediğin türü bul
    top_genre = "|".join(user_full['genres']).split("|")
    favorite_genre = pd.Series(top_genre).value_counts().index[0]
    
    watched = user_full['movieId'].tolist()
    recoms = movies_df[(movies_df['genres'].str.contains(favorite_genre)) & (~movies_df['movieId'].isin(watched))]
    
    return recoms.sort_values('success_rate', ascending=False).head(5), favorite_genre

# ==========================================
# 🎨 3. GÖRSEL ARAYÜZ (FRONTEND)
# ==========================================

st.set_page_config(page_title="Movie Lab v2.0", layout="wide")
movies, ratings = load_and_process_data()

# SIDEBAR: KONTROL PANELİ
st.sidebar.title("🔬 Proje Kontrol Merkezi")
u_id = st.sidebar.selectbox("Kullanıcı Seçiniz:", sorted(ratings['userId'].unique()), index=17)

# KULLANICI ÖZETİ
user_data = ratings[ratings['userId'] == u_id].merge(movies, on='movieId')
st.title(f"🎬 Analiz Raporu: Kullanıcı #{u_id}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("İzlenen Film", len(user_data))
c2.metric("Favoriler (5★)", len(user_data[user_data['rating'] == 5]))
c3.metric("Kişisel Memnuniyet", f"%{(user_data['rating'].mean()*20):.1f}")
c4.metric("İzleme Kalitesi", f"%{user_data['success_rate'].mean():.1f}")

st.divider()

# SEKMELİ ANALİZ (Hocanın İstediği Ayrı İnceleme)
tab1, tab2, tab3 = st.tabs(["👥 User-Based", "🎬 Item-Based", "🧬 Content-Based"])

with tab1:
    st.subheader("Kullanıcı Tabanlı Filtreleme (Sosyal Benzerlik)")
    st.info("Mantık: Sizinle aynı filmlere benzer puanlar veren kişilerin diğer favorilerini bulur.")
    try:
        ub_res, s_id, s_score = get_user_based(u_id, ratings, movies)
        st.write(f"**Tespit Edilen Ruh İkizi:** Kullanıcı {s_id} (Benzerlik: %{s_score*100:.1f})")
        cols = st.columns(5)
        for i, r in enumerate(ub_res.iterrows()):
            with cols[i]:
                st.success(f"**{r[1]['title']}**")
                st.caption(f"Başarı: %{r[1]['success_rate']}")
                st.progress(r[1]['success_rate']/100)
    except: st.warning("Benzerlik hesaplanamadı.")

with tab2:
    st.subheader("Ürün Tabanlı Filtreleme (Davranışsal Benzerlik)")
    st.info("Mantık: 5 yıldız verdiğiniz filmleri seven 'diğer insanların' en çok ortaklaştığı yapımlar.")
    ib_res = get_item_based(u_id, ratings, movies)
    if not ib_res.empty:
        cols = st.columns(5)
        for i, r in enumerate(ib_res.iterrows()):
            with cols[i]:
                st.warning(f"**{r[1]['title']}**")
                st.caption(f"Eşleşme: {r[1]['match_count']} Kişi")
                st.progress(r[1]['success_rate']/100)

with tab3:
    st.subheader("İçerik Tabanlı Filtreleme (Tür Analizi)")
    st.info("Mantık: Geçmişinizde en çok tercih ettiğiniz film türlerindeki en yüksek puanlı yapımlar.")
    cb_res, genre = get_content_based(u_id, ratings, movies)
    st.write(f"**Baskın İlgi Alanınız:** {genre}")
    cols = st.columns(5)
    for i, r in enumerate(cb_res.iterrows()):
        with cols[i]:
            st.info(f"**{r[1]['title']}**")
            st.caption(r[1]['genres'])
            st.progress(r[1]['success_rate']/100)

# KARAKTERİSTİK ANALİZ (Bonus Görsel)
st.divider()
st.header("🧬 Karakteristik Favorileriniz")
st.write("Genel beğeninin aksine, sizin şahsen çok daha yüksek değer verdiğiniz 'size özel' keşifler:")
user_data['diff'] = (user_data['rating'] * 20) - user_data['success_rate']
char_favs = user_data.sort_values('diff', ascending=False).head(5)
cols_c = st.columns(5)
for i, r in enumerate(char_favs.iterrows()):
    with cols_c[i]:
        st.write(f"**{r[1]['title']}**")
        st.write(f"Sizin: {r[1]['rating']}⭐ | Genel: %{r[1]['success_rate']}")