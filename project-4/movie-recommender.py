import numpy as np
import pandas as pd
import streamlit as st
from streamlit_star_rating import st_star_rating

import warnings
warnings.filterwarnings('ignore')

st.set_page_config(layout="wide")

@st.cache_data
def load_data():
	base_folder = "project-4"
	base_folder = "."
	ratings = pd.read_csv(f'{base_folder}/ratings.dat', sep='::', engine = 'python', header=None)
	ratings.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']
	
	movies = pd.read_csv(f'{base_folder}/movies.dat', sep='::', engine = 'python',
	                     encoding="ISO-8859-1", header = None)
	movies.columns = ['MovieID', 'Title', 'Genres']

	users = pd.read_csv(f'{base_folder}/users.dat', sep='::', engine = 'python', header = None)
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

@st.cache_data
def get_random_movie_set(n=10):
    movie_set = movies.sample(n)
    return movie_set

def get_recommendations():
    movie_set["star"] = np.array(star_list)


    reco_movies = movies.sample(n=reco_size)
    #print(movie_set)
    #print(star_list)
    return reco_movies

(ratings, movies, users) = load_data()
(genre_movie_ratings) = build_movie_recommendation_model_by_genre()



st.header("Movie Recommender System")
tab1, tab2 = st.tabs(["Genre", "Collaborative Filtering"])

with tab1:
	selected_genre = st.selectbox('Select a Genre', get_all_genre())
	movie_count = st.slider("Number of Recommendations:", 1, 100, 10)

	grid_size = st.slider("Display Grid:", 1, 10, 5)

	top_movies = find_top_movies_by_genre(genre=selected_genre, n=movie_count)

	#st.dataframe(top_movies)
	#Display selected Movies
	cols = st.columns(grid_size)

	(row, _) = top_movies.shape
	for i in range(row):
		record = top_movies.iloc[i, :]
		with cols[i%grid_size]:
			title = record['Title']
			st.subheader(f"{title}")

			st.text(f"Rank: {i+1}")

			weighted_rating =  f"{np.round(record['Weighted_Rating'], 2)} ⭐"
			st.text(f"Weighted Rating: {weighted_rating}")

			rating =  f"{np.round(record['Rating'], 2)} ⭐"
			st.text(f"Rating: {rating}")

			rating_count =  f"{int(record['Rating_count'])}"
			st.text(f"Rating Count: {rating_count}")

			image_url = f'https://liangfgithub.github.io/MovieImages/{record["MovieID"]}.jpg'
			st.image(image_url)

			st.divider()


with tab2:
	movie_set_size = st.slider("Movie Set size:", 1, 100, 10)
	reco_size = st.slider("Recommendation Set size:", 1, 100, 10)
	grid_size = st.slider("Display Grid:", 1, 10, 5, key="ibcf_grid_size")

	with st.container(border=True):
		with st.expander("Step 1: Rate as many movies as possible", expanded=True):
			st.info("Step 1: Rate as many movies as possible")
			movie_set = get_random_movie_set(n=movie_set_size)
			#movie_set["star"] = None

			cols = st.columns(grid_size)

			# Show Movie Set for User Rating
			star_list =list()
			(row, _) = movie_set.shape
			for i in range(row):
				record = movie_set.iloc[i, :]
				with cols[i % grid_size]:
					title = record['Title']
					st.subheader(f"{title}")

					image_url = f'https://liangfgithub.github.io/MovieImages/{record["MovieID"]}.jpg'
					st.image(image_url)

					star = st_star_rating("Give your rating", maxValue=5, defaultValue=0, key=f"stars_{i}")
					star_list.append(star)
					#record["xyz"] = star
					#st.write(star)

					#st.divider()

	with st.container(border=True):
		st.info("Step 2: Discover movies you might like")
		st.button("Get Recommendations", type="primary")

		#st.dataframe(movie_set)
		#st.write(movie_set)

		reco_movies = get_recommendations()
		(row, _) = reco_movies.shape

		cols = st.columns(grid_size)

		for i in range(row):
			record = reco_movies.iloc[i, :]
			with cols[i%grid_size]:
				title = record['Title']
				st.subheader(f"{title}")

				st.text(f"Rank: {i+1}")

				image_url = f'https://liangfgithub.github.io/MovieImages/{record["MovieID"]}.jpg'
				st.image(image_url)

				#st.divider()		



