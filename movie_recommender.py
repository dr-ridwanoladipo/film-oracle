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

st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stButton>button {
        color: #ffffff;
        background-color: #ff4b4b;
        border-radius: 5px;
    }
    .stSelectbox {
        color: #ffffff;
    }
    .movie-container {
        background-color: #1e2130;
        padding: 20px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_recommender():
    return MovieRecommender('movies.csv', 'ratings.csv')


def main():
    recommender = load_recommender()

    st.title('🎬 Movie Recommendation System')

    if 'page' not in st.session_state:
        st.session_state.page = "Home"
    if 'genre' not in st.session_state:
        st.session_state.genre = None

    pages = ["Home", "Popular Movies", "Content-based Recommendations", "Collaborative Filtering"]
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", pages, index=pages.index(st.session_state.page))

    if page != st.session_state.page:
        st.session_state.page = page
        if page != "Popular Movies":
            st.session_state.genre = None

    if st.session_state.page == "Home":
        show_home_page(recommender)
    elif st.session_state.page == "Popular Movies":
        show_popular_movies(recommender)
    elif st.session_state.page == "Content-based Recommendations":
        show_content_based(recommender)
    elif st.session_state.page == "Collaborative Filtering":
        show_collaborative_filtering(recommender)


def show_home_page(recommender):
    st.header("Welcome to the Movie Recommender!")
    st.write("Click on a genre to see popular movies in that category.")

    cols = st.columns(5)
    for i, genre in enumerate(recommender.genres):
        if cols[i % 5].button(genre):
            st.session_state.genre = genre
            st.session_state.page = "Popular Movies"
            st.rerun()


def show_popular_movies(recommender):
    st.header('Popular Movies')

    genre = st.selectbox('Select a genre', ['All'] + recommender.genres,
                         index=0 if st.session_state.genre is None else recommender.genres.index(
                             st.session_state.genre) + 1)
    n_movies = st.number_input('Number of movies to show', min_value=1, max_value=20, value=10)

    popular_movies = recommender.popularity_based_recommendations(n_movies, genre)

    for _, movie in popular_movies.iterrows():
        with st.container():
            col1, col2 = st.columns([1, 3])
            with col1:
                st.image(f"https://via.placeholder.com/150x225.png?text={movie['title'].replace(' ', '+')}",
                         use_column_width=True)
            with col2:
                st.subheader(movie['title'])
                st.write(f"**Year:** {movie['year']}")
                st.write(f"**Genres:** {', '.join([g['name'] for g in movie['genres']])}")
                st.write(f"**Rating:** {movie['weighted_rating']:.2f}")


def show_content_based(recommender):
    st.header('Content-based Recommendations')
    movie_title = st.selectbox('Select a movie', recommender.movies['title'].unique())
    n_recommendations = st.slider('Number of recommendations', 1, 10, 3)

    if st.button('Get Recommendations'):
        recommendations = recommender.content_based_recommendations(movie_title, n_recommendations)

        for _, movie in recommendations.iterrows():
            with st.container():
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.image(f"https://via.placeholder.com/150x225.png?text={movie['title'].replace(' ', '+')}",
                             use_column_width=True)
                with col2:
                    st.subheader(movie['title'])
                    st.write(f"**Year:** {movie['year']}")
                    st.write(f"**Genres:** {', '.join([g['name'] for g in movie['genres']])}")
                    st.write(f"**Overview:** {movie['overview'][:200]}...")


def show_collaborative_filtering(recommender):
    st.header('Collaborative Filtering Recommendations')
    user_id = st.number_input('Enter User ID', min_value=1, max_value=610, value=1)
    n_recommendations = st.slider('Number of recommendations', 1, 10, 3)

    if st.button('Get Recommendations'):
        recommendations = recommender.collaborative_filtering_recommendations(user_id, n_recommendations)

        for _, movie in recommendations.iterrows():
            with st.container():
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.image(f"https://via.placeholder.com/150x225.png?text={movie['title'].replace(' ', '+')}",
                             use_column_width=True)
                with col2:
                    st.subheader(movie['title'])
                    st.write(f"**Year:** {movie['year']}")
                    st.write(f"**Genres:** {', '.join([g['name'] for g in movie['genres']])}")
                    st.write(f"**Overview:** {movie['overview'][:200]}...")


if __name__ == '__main__':
    main()