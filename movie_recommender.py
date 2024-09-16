import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import streamlit as st


class MovieRecommender:
    def __init__(self, movies_path, ratings_path):
        """
        Initialize the MovieRecommender with data files.

        :param movies_path: Path to the movies CSV file
        :param ratings_path: Path to the ratings CSV file
        """
        self.movies = pd.read_csv(movies_path)
        self.ratings = pd.read_csv(ratings_path)
        self.prepare_data()

    def prepare_data(self):
        """Prepare and clean the data for analysis."""
        self.movies['overview'] = self.movies['overview'].fillna('')
        self.ratings = self.ratings[['userId', 'movieId', 'rating']]
        # Ensure movieId in ratings matches id in movies
        self.movies['movieId'] = self.movies['id']

    def popularity_based_recommendations(self, n=10):
        """
        Generate popularity-based movie recommendations.

        :param n: Number of recommendations to return
        :return: DataFrame of top N movies based on weighted rating
        """
        m = self.movies['vote_count'].quantile(0.9)
        C = self.movies['vote_average'].mean()

        qualified = self.movies.copy().loc[self.movies['vote_count'] >= m]
        qualified['weighted_rating'] = qualified.apply(
            lambda x: (x['vote_count'] / (x['vote_count'] + m) * x['vote_average']) +
                      (m / (m + x['vote_count']) * C),
            axis=1
        )

        return qualified.sort_values('weighted_rating', ascending=False)[['title', 'weighted_rating']].head(n)

    def content_based_recommendations(self, movie_title, n=3):
        """
        Generate content-based movie recommendations.

        :param movie_title: Title of the movie to base recommendations on
        :param n: Number of recommendations to return
        :return: List of recommended movie titles
        """
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(self.movies['overview'])

        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

        idx = self.movies[self.movies['title'] == movie_title].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:n + 1]
        movie_indices = [i[0] for i in sim_scores]

        return list(self.movies['title'].iloc[movie_indices])

    def collaborative_filtering_recommendations(self, user_id, n=3):
        """
        Generate collaborative filtering-based movie recommendations.

        :param user_id: ID of the user to generate recommendations for
        :param n: Number of recommendations to return
        :return: List of recommended movie titles
        """
        # Create a user-item matrix
        user_item_matrix = self.ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

        # Compute cosine similarity between users
        user_similarity = cosine_similarity(user_item_matrix)

        # Get similar users
        similar_users = user_similarity[user_id - 1].argsort()[::-1][1:11]  # top 10 similar users

        # Get movies rated by similar users but not by the target user
        user_movies = set(self.ratings[self.ratings['userId'] == user_id]['movieId'])
        candidate_movies = set(self.ratings[self.ratings['userId'].isin(similar_users + 1)]['movieId']) - user_movies

        # Calculate predicted ratings for candidate movies
        predicted_ratings = []
        for movie in candidate_movies:
            avg_rating = self.ratings[self.ratings['movieId'] == movie]['rating'].mean()
            predicted_ratings.append((movie, avg_rating))

        # Sort and get top N recommendations
        recommended_movies = sorted(predicted_ratings, key=lambda x: x[1], reverse=True)

        # Get movie titles, ensuring they exist in the dataset
        movie_titles = []
        for movie_id, _ in recommended_movies:
            title = self.movies[self.movies['movieId'] == movie_id]['title']
            if not title.empty:
                movie_titles.append(title.iloc[0])

            if len(movie_titles) == n:
                break

        return movie_titles

def main():
    st.title('Movie Recommendation System')

    recommender = MovieRecommender('movies.csv', 'ratings.csv')

    st.header('Popularity-based Recommendations')
    n_popular = st.slider('Number of popular movies to show', 5, 20, 10)
    popular_movies = recommender.popularity_based_recommendations(n_popular)
    st.table(popular_movies)

    st.header('Content-based Recommendations')
    movie_title = st.selectbox('Select a movie', recommender.movies['title'].unique())
    n_content = st.slider('Number of content-based recommendations', 1, 10, 3)
    if st.button('Get Content-based Recommendations'):
        content_recommendations = recommender.content_based_recommendations(movie_title, n_content)
        st.write(content_recommendations)

    st.header('Collaborative Filtering Recommendations')
    user_id = st.number_input('Enter User ID', min_value=1, max_value=610, value=1)
    n_collab = st.slider('Number of collaborative filtering recommendations', 1, 10, 3)
    if st.button('Get Collaborative Filtering Recommendations'):
        collab_recommendations = recommender.collaborative_filtering_recommendations(user_id, n_collab)
        st.write(collab_recommendations)


if __name__ == '__main__':
    main()