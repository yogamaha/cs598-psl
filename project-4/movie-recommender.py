import numpy as np
import pandas as pd
import streamlit as st

import warnings
warnings.filterwarnings('ignore')

@st.cache_data
def load_data():
	ratings = pd.read_csv('project-4/ratings.dat', sep='::', engine = 'python', header=None)
	ratings.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']
	
	movies = pd.read_csv('project-4/movies.dat', sep='::', engine = 'python',
	                     encoding="ISO-8859-1", header = None)
	movies.columns = ['MovieID', 'Title', 'Genres']

	users = pd.read_csv('project-4/users.dat', sep='::', engine = 'python', header = None)
	users.columns = ['UserID', 'Gender', 'Age', 'Occupation', 'Zipcode']

	return (ratings, movies, users)

@st.cache_data
def build_movie_recommendation_model_by_genre():
	rating_merged = ratings.merge(movies, left_on = 'MovieID', right_on = 'MovieID')

	movie_rating = rating_merged[['MovieID', 'Rating']].groupby("MovieID").agg(['mean', 'count']).droplevel(0, axis=1).reset_index()

	movie_rating.rename(columns={"mean": "Rating", "count": "Rating_count"}, inplace=True)


	avg_rating_count = movie_rating['Rating_count'].mean() 
	avg_rating = movie_rating['Rating'].min()

	movie_rating['Weighted_Rating'] = (movie_rating['Rating'] * movie_rating['Rating_count'] + avg_rating * avg_rating_count)  / (movie_rating['Rating_count'] + avg_rating_count)

	movie_with_rating = movies.join(movie_rating.set_index('MovieID'), how='left', on="MovieID")

	movie_with_rating['Weighted_Rating'].fillna(value=avg_rating, inplace=True)

	genre_movie_ratings = movie_with_rating.copy()
	genre_movie_ratings['Genres'] = genre_movie_ratings['Genres'].str.split('|')
	genre_movie_ratings = genre_movie_ratings.explode('Genres')

	return (genre_movie_ratings)

@st.cache_data
def get_all_genre():
    genres = genre_movie_ratings['Genres'].unique()
    return genres

@st.cache_data
def find_top_movies_by_genre(genre, n=10):
    top_movies = genre_movie_ratings[genre_movie_ratings['Genres'] == genre]
    top_movies = top_movies.sort_values(by='Weighted_Rating', ascending=False)    
    top_movies = top_movies[0:n]
    return top_movies

def display_movies(movies_df):
	grid_size = 2
	cols = st.columns(grid_size)

	(row, _) = movies_df.shape
	for i in range(row):
		record = movies_df.iloc[i, :]
		with cols[i%grid_size]:
			title = record['Title']
			st.subheader(f"{title}")

			weighted_rating =  f"{np.round(record['Weighted_Rating'], 2)} ⭐"
			st.text(f"Weighted Rating: {weighted_rating}")

			rating =  f"{np.round(record['Rating'], 2)} ⭐"
			st.text(f"Rating: {rating}")

			rating_count =  f"{int(record['Rating_count'])}"
			st.text(f"Rating Count: {rating_count}")

			image_url = f'https://liangfgithub.github.io/MovieImages/{record["MovieID"]}.jpg'
			st.image(image_url)

			st.divider()

	return


	


(ratings, movies, users) = load_data()
(genre_movie_ratings) = build_movie_recommendation_model_by_genre()


st.header("Movie Recommender System")
tab1, tab2 = st.tabs(["Genre", "Collaborative Filtering"])

with tab1:
	selected_genre = st.selectbox('Select a Genre', get_all_genre())
	movie_count = st.slider("Number of Recommendations:", 1, 100, 10)

	top_movies = find_top_movies_by_genre(genre=selected_genre, n=movie_count)
	#st.dataframe(top_movies)
	display_movies(top_movies)

