import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.preprocessing import MinMaxScaler


class MovieRecommender:
    def __init__(self, movies_path, ratings_path):
        self.movies = pd.read_csv(movies_path)
        self.ratings = pd.read_csv(ratings_path)
        self.prepare_data()

    def prepare_data(self):
        self.movies['overview'] = self.movies['overview'].fillna('')
        self.ratings = self.ratings[['userId', 'movieId', 'rating']]
        self.movies['movieId'] = self.movies['id']
        self.movies['year'] = pd.to_datetime(self.movies['release_date']).dt.year
        self.movies['genres'] = self.movies['genres'].apply(eval)
        self.genres = sorted(set([genre['name'] for genres in self.movies['genres'] for genre in genres]))

    def popularity_based_recommendations(self, n=10, genre=None):
        qualified = self.movies.copy()

        if genre and genre != 'All':
            qualified = qualified[qualified['genres'].apply(lambda x: genre in [g['name'] for g in x])]

        m = qualified['vote_count'].quantile(0.9)
        C = qualified['vote_average'].mean()

        qualified['weighted_rating'] = qualified.apply(
            lambda x: (x['vote_count'] / (x['vote_count'] + m) * x['vote_average']) +
                      (m / (m + x['vote_count']) * C),
            axis=1
        )

        return qualified.sort_values('weighted_rating', ascending=False)[
            ['title', 'weighted_rating', 'year', 'genres']].head(n)

    def content_based_recommendations(self, movie_title, n=3):
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(self.movies['overview'])
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        idx = self.movies[self.movies['title'] == movie_title].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:n + 1]
        movie_indices = [i[0] for i in sim_scores]
        return self.movies.iloc[movie_indices][['title', 'year', 'genres', 'overview']]

    def collaborative_filtering_recommendations(self, user_id, n=3):
        user_item_matrix = self.ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
        user_similarity = cosine_similarity(user_item_matrix)
        similar_users = user_similarity[user_id - 1].argsort()[::-1][1:11]
        user_movies = set(self.ratings[self.ratings['userId'] == user_id]['movieId'])
        candidate_movies = set(self.ratings[self.ratings['userId'].isin(similar_users + 1)]['movieId']) - user_movies
        predicted_ratings = []
        for movie in candidate_movies:
            avg_rating = self.ratings[self.ratings['movieId'] == movie]['rating'].mean()
            predicted_ratings.append((movie, avg_rating))
        recommended_movies = sorted(predicted_ratings, key=lambda x: x[1], reverse=True)[:n]
        return self.movies[self.movies['movieId'].isin([movie for movie, _ in recommended_movies])][
            ['title', 'year', 'genres', 'overview']]


# Streamlit App
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(45deg, #FF512F, #DD2476, #FF512F);
        background-size: 200% 200%;
        animation: gradient 15s ease infinite;
        color: #ffffff;
    }
    @keyframes gradient {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    .stButton>button {
        color: #ffffff;
        background-color: #4A148C;
        border-radius: 5px;
        font-weight: bold;
        border: none;
        padding: 0.5rem 1rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #7B1FA2;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stSelectbox>div>div {
        background-color: rgba(255,255,255,0.2);
        color: #ffffff;
        border-radius: 5px;
    }
    .movie-container {
        background-color: rgba(0,0,0,0.5);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
    }
    .st-expander {
        background-color: rgba(0,0,0,0.5);
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.2);
        backdrop-filter: blur(10px);
    }
    .st-expander .st-expander-content {
        background-color: transparent;
    }
    h1, h2, h3 {
        color: #FFD700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    .stTextInput>div>div>input {
        color: #ffffff;
        background-color: rgba(255,255,255,0.2);
        border-radius: 5px;
    }
    .hamburger {
        font-size: 24px;
        cursor: pointer;
        background-color: #4A148C;
        color: #ffffff;
        padding: 5px 10px;
        border-radius: 5px;
        display: inline-block;
        transition: all 0.3s ease;
    }
    .hamburger:hover {
        background-color: #7B1FA2;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
        .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: white;
        color: #2c3e50;
        text-align: center;
        padding: 10px 0;
        font-weight: bold;
    }

</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_recommender():
    return MovieRecommender('movies.csv', 'ratings.csv')


def main():
    recommender = load_recommender()

    if 'page' not in st.session_state:
        st.session_state.page = "Home"
    if 'genre' not in st.session_state:
        st.session_state.genre = None
    if 'trigger_popular_movies' not in st.session_state:
        st.session_state.trigger_popular_movies = False
    if 'recommendation_page' not in st.session_state:
        st.session_state.recommendation_page = 0
    if 'current_recommendations' not in st.session_state:
        st.session_state.current_recommendations = None

    # Hamburger menu
    with st.expander("☰ Menu"):
        pages = ["Home", "Popular Movies", "Content-based Recommendations", "Collaborative Filtering"]
        for page in pages:
            if st.button(page, key=f"nav_{page}"):
                st.session_state.page = page
                if page != "Popular Movies":
                    st.session_state.genre = None
                    st.session_state.trigger_popular_movies = False
                st.session_state.recommendation_page = 0
                st.session_state.current_recommendations = None
                st.rerun()

    st.title('🎬 Movie Recommendation System')

    if st.session_state.trigger_popular_movies:
        st.session_state.page = "Popular Movies"
        st.session_state.trigger_popular_movies = False

    if st.session_state.page == "Home":
        show_home_page(recommender)
    elif st.session_state.page == "Popular Movies":
        show_popular_movies(recommender)
    elif st.session_state.page == "Content-based Recommendations":
        show_content_based(recommender)
    elif st.session_state.page == "Collaborative Filtering":
        show_collaborative_filtering(recommender)

    st.markdown(
        """
        <div class="footer">
        © 2024 All Rights Reserved | Dr. Ridwan Oladipo
        </div>
        """,
        unsafe_allow_html=True
    )


def show_home_page(recommender):
    st.header("Welcome to the Movie Recommender!")
    st.write("Click on a genre to see popular movies in that category.")

    cols = st.columns(5)
    for i, genre in enumerate(recommender.genres):
        if cols[i % 5].button(genre, key=f"genre_{genre}"):
            st.session_state.genre = genre
            st.session_state.trigger_popular_movies = True
            st.session_state.recommendation_page = 0
            st.session_state.current_recommendations = None
            st.rerun()


def display_movie(movie):
    with st.container():
        st.subheader(movie['title'])
        st.write(f"**Year:** {movie['year']}")
        st.write(f"**Genres:** {', '.join([g['name'] for g in movie['genres']])}")
        if 'weighted_rating' in movie:
            st.write(f"**Rating:** {movie['weighted_rating']:.2f}")
        if 'overview' in movie:
            st.write(f"**Overview:** {movie['overview'][:200]}...")
    st.write("")  # Add some space between movies


def show_popular_movies(recommender):
    st.header('Popular Movies')

    genres = ['All'] + recommender.genres
    if st.session_state.genre is None or st.session_state.genre not in genres:
        selected_index = 0
    else:
        selected_index = genres.index(st.session_state.genre)

    genre = st.selectbox('Select a genre', genres,
                         index=selected_index,
                         key="genre_selectbox")

    if genre != st.session_state.genre:
        st.session_state.genre = genre
        st.session_state.recommendation_page = 0
        st.session_state.current_recommendations = None

    n_movies = st.number_input('Number of movies to show', min_value=1, max_value=20, value=10)

    if st.session_state.current_recommendations is None:
        st.session_state.current_recommendations = recommender.popularity_based_recommendations(100,
                                                                                                genre if genre != 'All' else None)

    start_index = st.session_state.recommendation_page * n_movies
    end_index = start_index + n_movies

    for _, movie in st.session_state.current_recommendations.iloc[start_index:end_index].iterrows():
        display_movie(movie)

    col1, col2 = st.columns(2)
    with col1:
        if st.button('Previous', disabled=(st.session_state.recommendation_page == 0)):
            st.session_state.recommendation_page -= 1
            st.rerun()
    with col2:
        if st.button('Next', disabled=(end_index >= len(st.session_state.current_recommendations))):
            st.session_state.recommendation_page += 1
            st.rerun()


def show_content_based(recommender):
    st.header('Content-based Recommendations')
    movie_title = st.selectbox('Select a movie', recommender.movies['title'].unique())
    n_recommendations = st.slider('Number of recommendations', 1, 10, 3)

    if st.button('Get Recommendations') or st.session_state.current_recommendations is not None:
        if st.session_state.current_recommendations is None:
            st.session_state.current_recommendations = recommender.content_based_recommendations(movie_title, 100)

        start_index = st.session_state.recommendation_page * n_recommendations
        end_index = start_index + n_recommendations

        for _, movie in st.session_state.current_recommendations.iloc[start_index:end_index].iterrows():
            display_movie(movie)

        col1, col2 = st.columns(2)
        with col1:
            if st.button('Previous', disabled=(st.session_state.recommendation_page == 0)):
                st.session_state.recommendation_page -= 1
                st.rerun()
        with col2:
            if st.button('Next', disabled=(end_index >= len(st.session_state.current_recommendations))):
                st.session_state.recommendation_page += 1
                st.rerun()


def show_collaborative_filtering(recommender):
    st.header('Collaborative Filtering Recommendations')
    user_id = st.number_input('Enter User ID', min_value=1, max_value=610, value=1)
    n_recommendations = st.slider('Number of recommendations', 1, 10, 3)

    if st.button('Get Recommendations') or st.session_state.current_recommendations is not None:
        if st.session_state.current_recommendations is None:
            st.session_state.current_recommendations = recommender.collaborative_filtering_recommendations(user_id, 100)

        start_index = st.session_state.recommendation_page * n_recommendations
        end_index = start_index + n_recommendations

        for _, movie in st.session_state.current_recommendations.iloc[start_index:end_index].iterrows():
            display_movie(movie)

        col1, col2 = st.columns(2)
        with col1:
            if st.button('Previous', disabled=(st.session_state.recommendation_page == 0)):
                st.session_state.recommendation_page -= 1
                st.rerun()
        with col2:
            if st.button('Next', disabled=(end_index >= len(st.session_state.current_recommendations))):
                st.session_state.recommendation_page += 1
                st.rerun()


if __name__ == '__main__':
    main()