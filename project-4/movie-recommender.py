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
	#base_folder = "."
	ratings = pd.read_csv(f'{base_folder}/ratings.dat', sep='::', engine = 'python', header=None)
	ratings.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']
	
	movies = pd.read_csv(f'{base_folder}/movies.dat', sep='::', engine = 'python',
	                     encoding="ISO-8859-1", header = None)
	movies.columns = ['MovieID', 'Title', 'Genres']

	users = pd.read_csv(f'{base_folder}/users.dat', sep='::', engine = 'python', header = None)
	users.columns = ['UserID', 'Gender', 'Age', 'Occupation', 'Zipcode']

	#rating_matrix = ratings.pivot_table(index="UserID", columns="MovieID", values="Rating")
	rating_matrix = pd.read_csv(f'{base_folder}/Rmat.csv', sep=',')

	return (ratings, movies, users, rating_matrix)

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

@st.cache_data
def build_similarity_matrix_v2():
    normalized_rating_matrix = rating_matrix.subtract(rating_matrix.mean(axis=1), axis='rows')

    cardinality_df = (~normalized_rating_matrix.isna()).astype('int')
    cardinality_df = cardinality_df.T
    cardinality_matrix = cardinality_df @ cardinality_df.T
    
    normalized_rating_matrix = normalized_rating_matrix.T
    normalized_rating_matrix = normalized_rating_matrix.fillna(0)

    nr = normalized_rating_matrix @ normalized_rating_matrix.T
    #print(nr)

    squared_normalized_rating_matrix = ((normalized_rating_matrix**2) @ (normalized_rating_matrix!=0).T)
    squared_normalized_rating_matrix = squared_normalized_rating_matrix.apply(np.vectorize(np.sqrt))
    dr = squared_normalized_rating_matrix * squared_normalized_rating_matrix.T
    #print(dr)
    
    cosine_distance = nr/dr
    S = (1 + cosine_distance)/2
    #print(S)
    
    np.fill_diagonal(S.values, np.nan)

    S[cardinality_matrix<3] = None
    #print(S)
    
    #S[S.rank(axis=1, ascending=False)>30] = None
    #print(S)
    return S

def myIBCF(S, w, n=10):
    S = S.copy()
    S = S.fillna(0)

    w = w.copy()
    identity = (~w.isna()).astype(int)
    w = w.fillna(0)

    reco_movies = w.dot(S) / identity.dot(S)
    reco_movies = reco_movies.sort_values(ascending=False)[0:n]
    
    reco_movies = reco_movies.dropna()
    
    if reco_movies.size < n:
        print("Backfilling from Genre based recommendations")        
        backfill_count = n - reco_movies.size
        random_genre = np.random.choice(get_all_genre())
        backfill_df = find_top_movies_by_genre(genre=random_genre, n=backfill_count)
        
        backfill_movies = pd.Series(data=backfill_df["Weighted_Rating"].values, 
                                    index=("m" +backfill_df["MovieID"].astype(str)).values)
        reco_movies = pd.concat([reco_movies, backfill_movies], axis=0)
    
    return reco_movies

def get_recommendations():
    movie_set["star"] = np.array(star_list)

    row = S.iloc[0]
    user_ratings = row.copy()
    user_ratings[:] = np.nan

    for i in range(movie_set.shape[0]):
    	key = "m" + str(movie_set.iloc[i]["MovieID"])
    	value = movie_set.iloc[i]["star"]
    	if key in user_ratings:
    		user_ratings.loc[key] = value

    #print(user_ratings.dropna())
    #print(movie_set)
    #print(star_list)

    reco_movies = myIBCF(S=S, w=user_ratings, n=reco_size)
    #print("myIBCF-reco_movies")
    #print(reco_movies)
    reco_movies = movies[movies["MovieID"].isin(reco_movies.index.str.slice(1).astype(int))]
    #reco_movies = movies.sample(n=reco_size)

    return reco_movies

(ratings, movies, users, rating_matrix) = load_data()
(genre_movie_ratings) = build_movie_recommendation_model_by_genre()
S = build_similarity_matrix_v2()


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



