import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
#  BACKEND (HESAPLAMA MOTORU)
# ==========================================

@st.cache_data
def load_data():
    # Veri setlerini oku
    movies = pd.read_csv('movies.csv')
    ratings = pd.read_csv('ratings.csv')
    
    # Sütun temizliği
    movies.columns = [c.strip() for c in movies.columns]
    ratings.columns = [c.strip() for c in ratings.columns]

    # Başarı Oranı ve Popülerlik Analizi
    # Ortalama puanı alıp % formatına çeviriyoruz
    movie_stats = ratings.groupby('movieId')['rating'].agg(['mean', 'count']).reset_index()
    movie_stats['success_rate'] = (movie_stats['mean'] / 5 * 100).round(1)
    movies = movies.merge(movie_stats, on='movieId', how='left').fillna(0)
    
    return movies, ratings

def get_user_profile(user_id, ratings_df, movies_df):
    """Kullanıcının geçmişini ve karakterini analiz eder"""
    user_ratings = ratings_df[ratings_df['userId'] == user_id]
    user_full_data = user_ratings.merge(movies_df, on='movieId')
    
    # Hipsterlık Skoru (İzlediği filmlerin ortalama oylanma sayısı)
    avg_pop = user_full_data['count'].mean()
    if avg_pop < 60:
        h_label = "Hipster "
    elif avg_pop < 100:
        h_label = "Dengeli "
    else:
        h_label = "Popüler "
        
    return user_full_data, h_label, avg_pop

def find_soulmate(target_user_id, ratings_df):
    """Kullanıcı bazlı işbirlikçi filtreleme ile zevk ikizini bulur"""
    # Performans için sadece 30'dan fazla oylanan filmleri matrise alalım
    popular_movies = ratings_df.groupby('movieId').size()[lambda x: x > 30].index
    filtered_ratings = ratings_df[ratings_df['movieId'].isin(popular_movies)]
    
    matrix = filtered_ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
    
    if target_user_id not in matrix.index:
        return None, 0
        
    # Cosine Similarity ile benzerlik matrisi
    sim = cosine_similarity(matrix)
    sim_df = pd.DataFrame(sim, index=matrix.index, columns=matrix.index)
    
    # En benzer kullanıcı (kendisi hariç)
    soulmate_id = sim_df[target_user_id].sort_values(ascending=False).index[1]
    similarity_score = sim_df.loc[target_user_id, soulmate_id]
    
    return soulmate_id, similarity_score

# ==========================================
#  FRONTEND (ARAYÜZ)
# ==========================================

st.set_page_config(page_title="Movie DNA Analysis", layout="wide")
movies, ratings = load_data()

# SIDEBAR
st.sidebar.title("👤 Kullanıcı Seçimi")
st.sidebar.markdown("Analiz etmek istediğiniz kullanıcıyı listeden seçin veya yazın.")

# 1'den 610'a kadar olan kullanıcı listesi
user_list = sorted(ratings['userId'].unique())
selected_user = st.sidebar.selectbox("Kullanıcı ID:", options=user_list, index=17) # Varsayılan User 18

if selected_user:
    user_data, hipster_label, pop_val = get_user_profile(selected_user, ratings, movies)
    
    st.title(f"📊 Kullanıcı #{selected_user} - Film Tercih Raporu")
    st.info(f"Bu analiz, kullanıcının puanladığı **{len(user_data)}** film verisi üzerinden hesaplanmıştır.")

    # 1. BÖLÜM: ÜST METRİKLER (GİRİFT ANALİZ)
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("İzleyici Tipi", hipster_label)
        st.caption(f"Popülerlik Endeksi: {int(pop_val)} (Düşük=Daha Özgün)")
    with m2:
        # Kullanıcının kendi verdiği puanların ortalamasını % yapalım
        avg_rating_pct = (user_data['rating'].mean() / 5 * 100)
        st.metric("Memnuniyet Oranı", f"%{avg_rating_pct:.1f}")
        st.caption("Verdiği puanların genel ortalaması")
    with m3:
        soulmate_id, score = find_soulmate(selected_user, ratings)
        st.metric("Zevk İkizi", f"User {soulmate_id}")
        st.caption(f"Zevk Benzerliği: %{score*100:.1f}")

    st.divider()

    # 2. BÖLÜM: GÖRSEL ANALİZLER
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.subheader(" Tür Dağılımı")
        # Türleri parçalayıp sayma
        genres_list = "|".join(user_data['genres']).split("|")
        genre_df = pd.DataFrame(genres_list, columns=['Tür']).value_counts().reset_index(name='Adet')
        fig_pie = px.pie(genre_df, values='Adet', names='Tür', hole=0.4, 
                         color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col_right:
        st.subheader("⭐ Karakterini Yansıtan Seçimler")
        # FARK ANALİZİ: Kendi puanı ile toplum başarısı arasındaki farkın en yüksek olduğu filmler
        # (Yani toplumun sıradan bulduğu ama onun bayıldığı filmler)
        favs = user_data[user_data['rating'] >= 4].copy()
        favs['diff'] = (favs['rating'] * 20) - favs['success_rate']
        
        # En "karakteristik" 5 film
        unique_choices = favs.sort_values('diff', ascending=False).head(5)
        
        for _, row in unique_choices.iterrows():
            st.write(f"**{row['title']}**")
            st.caption(f"Senin Puanın: {row['rating']} | Toplum Başarısı: %{row['success_rate']}")
            
            # Eğer toplumdan çok daha yüksek vermişse bilgi notu çıkar
            if row['diff'] > 15:
                st.info(f"💡 Bu senin gizli favorin! Toplumdan %{int(row['diff'])} daha fazla sevmişsin.")
            st.progress(row['success_rate']/100)

    # 3. BÖLÜM: TAVSİYELER
    st.divider()
    st.header(f" Zevk İkizinden (User {soulmate_id}) Sana Özel Öneriler")
    
    # Ruh ikizinin yüksek puan verdiği ama kullanıcının henüz izlemediği filmler
    soulmate_ratings = ratings[ratings['userId'] == soulmate_id]
    watched_ids = user_data['movieId'].tolist()
    
    recommendations = soulmate_ratings[(soulmate_ratings['rating'] >= 4) & (~soulmate_ratings['movieId'].isin(watched_ids))]
    recom_display = recommendations.merge(movies, on='movieId').sort_values('success_rate', ascending=False).head(3)

    if not recom_display.empty:
        cols = st.columns(3)
        for i, (_, row) in enumerate(recom_display.iterrows()):
            with cols[i]:
                st.success(f"**{row['title']}**")
                st.write(f" Tür: {row['genres']}")
                st.write(f" Toplum Puanı: %{row['success_rate']}")
                st.progress(row['success_rate']/100)
    else:
        st.write("Şu an için ruh ikizinden yeni bir öneri bulunmuyor.")