import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 🧠 1. VERİ VE HAZIRLIK
# ==========================================

@st.cache_data
def load_and_process_data():
    movies = pd.read_csv('movies.csv')
    ratings = pd.read_csv('ratings.csv')
    
    # Başarı Oranı Hesaplama
    movie_stats = ratings.groupby('movieId')['rating'].agg(['mean', 'count']).reset_index()
    movie_stats['success_rate'] = (movie_stats['mean'] / 5 * 100).round(1)
    
    movies = movies.merge(movie_stats, on='movieId', how='left').fillna(0)
    return movies, ratings

# ==========================================
# 🛠️ 2. ALGORİTMA FONKSİYONLARI
# ==========================================

def get_user_based(user_id, ratings_df, movies_df):
    pop_movies = ratings_df.groupby('movieId').size()[lambda x: x > 30].index
    matrix = ratings_df[ratings_df['movieId'].isin(pop_movies)].pivot_table(
        index='userId', columns='movieId', values='rating').fillna(0)
    
    sim = cosine_similarity(matrix)
    sim_df = pd.DataFrame(sim, index=matrix.index, columns=matrix.index)
    soulmate_id = sim_df[user_id].sort_values(ascending=False).index[1]
    sim_score = sim_df.loc[user_id, soulmate_id]
    
    watched = ratings_df[ratings_df['userId'] == user_id]['movieId'].tolist()
    recoms = ratings_df[(ratings_df['userId'] == soulmate_id) & (ratings_df['rating'] >= 4.5)]
    final = recoms[~recoms['movieId'].isin(watched)].merge(movies_df, on='movieId')
    return final.sort_values('success_rate', ascending=False).head(5), soulmate_id, sim_score

def get_item_based(user_id, ratings_df, movies_df):
    user_ratings = ratings_df[ratings_df['userId'] == user_id]
    top_films = user_ratings[user_ratings['rating'] == 5]
    if len(top_films) < 3:
        top_films = user_ratings[user_ratings['rating'] >= 4]
    
    seed_ids = top_films.sort_values('rating', ascending=False).head(5)['movieId'].tolist()
    recommendations = []
    for m_id in seed_ids:
        fans = ratings_df[(ratings_df['movieId'] == m_id) & (ratings_df['rating'] >= 4.5)]['userId'].unique()
        others = ratings_df[(ratings_df['userId'].isin(fans)) & (ratings_df['movieId'] != m_id)]
        recommendations.append(others)
    
    if not recommendations: return pd.DataFrame()
    combined = pd.concat(recommendations).groupby('movieId').size().reset_index(name='match_count')
    watched = user_ratings['movieId'].tolist()
    final = combined[~combined['movieId'].isin(watched)].merge(movies_df, on='movieId')
    return final.sort_values(['match_count', 'success_rate'], ascending=False).head(5)

def get_content_based(user_id, ratings_df, movies_df):
    user_full = ratings_df[ratings_df['userId'] == user_id].merge(movies_df, on='movieId')
    top_genres = "|".join(user_full['genres']).split("|")
    favorite_genre = pd.Series(top_genres).value_counts().index[0]
    watched = user_full['movieId'].tolist()
    recoms = movies_df[(movies_df['genres'].str.contains(favorite_genre)) & (~movies_df['movieId'].isin(watched))]
    return recoms.sort_values('success_rate', ascending=False).head(5), favorite_genre

# ==========================================
# 🎨 3. GÖRSEL ARAYÜZ (DASHBOARD)
# ==========================================

st.set_page_config(page_title="Algoritma Laboratuvarı", layout="wide")
movies, ratings = load_and_process_data()

st.sidebar.title("🧬 Kontrol Paneli")
u_id = st.sidebar.selectbox("Kullanıcı ID Seçin:", sorted(ratings['userId'].unique()), index=17)

# --- ÜST METRİKLER ---
user_data = ratings[ratings['userId'] == u_id].merge(movies, on='movieId')
st.title(f"📊 Kullanıcı #{u_id} Profil Analizi")

m1, m2, m3, m4 = st.columns(4)
m1.metric("İzlenen Film", len(user_data))
m2.metric("Favori (5★)", len(user_data[user_data['rating'] == 5]))
m3.metric("Memnuniyet", f"%{(user_data['rating'].mean()*20):.1f}")
m4.metric("Başarı Skoru", f"%{user_data['success_rate'].mean():.1f}")

st.divider()

# --- 🎯 O ÇOK İYİ DEDİĞİN GÖRSEL TABLO BÖLÜMÜ ---
col_left, col_right = st.columns([1, 1.2])

with col_left:
    st.subheader("🧬 Tür DNA'sı")
    all_genres = "|".join(user_data['genres']).split("|")
    genre_df = pd.DataFrame(all_genres, columns=['Tür'])
    fig = px.pie(genre_df, names='Tür', hole=0.4, 
                 color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("💎 Karakteristik Favoriler")
    st.caption("Genel beğeniden en çok ayrıştığınız, size özel zevkler:")
    user_data['diff'] = (user_data['rating'] * 20) - user_data['success_rate']
    char_favs = user_data.sort_values('diff', ascending=False).head(5)
    
    for _, row in char_favs.iterrows():
        st.write(f"**{row['title']}**")
        st.progress(row['success_rate'] / 100)
        st.caption(f"Sizin Puanınız: {row['rating']} ⭐ | Genel Başarı: %{row['success_rate']}")

st.divider()

# --- 🚀 ALGORİTMA SEKMELERİ ---
st.header("⚡ Akıllı Öneri Motorları")
tab1, tab2, tab3 = st.tabs(["👥 User-Based", "🎬 Item-Based", "🧬 Content-Based"])

with tab1:
    ub_res, s_id, s_score = get_user_based(u_id, ratings, movies)
    st.info(f"**Yöntem:** Ruh İkizi Analizi (Kullanıcı {s_id} ile %{s_score*100:.1f} uyum)")
    cols = st.columns(5)
    for i, r in enumerate(ub_res.iterrows()):
        with cols[i]:
            st.success(f"**{r[1]['title']}**")
            st.write(f"Başarı: %{r[1]['success_rate']}")

with tab2:
    st.info("**Yöntem:** Ürün Korelasyonu (5 yıldızlı favorilerinizin izinden)")
    ib_res = get_item_based(u_id, ratings, movies)
    if not ib_res.empty:
        cols = st.columns(5)
        for i, r in enumerate(ib_res.iterrows()):
            with cols[i]:
                st.warning(f"**{r[1]['title']}**")
                st.write(f"Eşleşme: {r[1]['match_count']}")

with tab3:
    cb_res, genre_name = get_content_based(u_id, ratings, movies)
    st.info(f"**Yöntem:** Tür DNA Eşleşmesi (En sevdiğiniz tür: {genre_name})")
    cols = st.columns(5)
    for i, r in enumerate(cb_res.iterrows()):
        with cols[i]:
            st.info(f"**{r[1]['title']}**")
            st.write(f"Tür: {r[1]['genres'].split('|')[0]}")