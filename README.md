# Movie Recommendation System

This project implements a comprehensive movie recommendation system using three different approaches: popularity-based filtering, content-based filtering, and collaborative filtering. It demonstrates the application of various data science and machine learning techniques in the field of recommendation systems.

## Features

- Popularity-based movie recommendations
- Content-based movie recommendations
- Collaborative filtering-based movie recommendations
- Streamlit web interface for easy interaction

## Requirements

- Python 3.12
- pandas
- numpy
- scikit-learn
- streamlit

## Installation

1. Clone this repository

2. Install the required packages:
   ```
   pip install pandas numpy scikit-learn streamlit
   ```

## Usage

1. Ensure you have the following CSV files in the project directory:
   - `movies.csv`: Contains movie information
   - `ratings.csv`: Contains user ratings for movies

2. Run the Streamlit app:
   ```
   streamlit run movie_recommender.py
   ```

3. The Streamlit app will open in your default web browser. You can interact with the different recommendation systems through the user interface.

## Streamlit Interface

The Streamlit interface provides three main sections:

1. Popularity-based Recommendations:
   - Use the slider to select the number of popular movies to display.

2. Content-based Recommendations:
   - Select a movie from the dropdown menu.
   - Use the slider to choose the number of recommendations.
   - Click the "Get Content-based Recommendations" button to see similar movies.

3. Collaborative Filtering Recommendations:
   - Enter a User ID.
   - Use the slider to select the number of recommendations.
   - Click the "Get Collaborative Filtering Recommendations" button to see personalized recommendations.

## Customization

You can modify the `MovieRecommender` class in `movie_recommender.py` to add more features or change existing ones. The Streamlit interface in the `main()` function can be adjusted to add more interactivity or display additional information.

## Contributing

Contributions, issues, and feature requests are welcome. Feel free to check the [issues page](https://github.com/yourusername/movie-recommendation-system/issues) if you want to contribute.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The MovieLens dataset for providing the movie data
- Streamlit for the easy-to-use web application framework