import streamlit as st
import pickle
import pandas as pd
import requests

def fetch_poster(movie_id):
   response = requests.get('https://api.themoviedb.org/3/movie/{}?api_key=5863fe291d506a6a83f02542eec644ca'.format(movie_id))
   data = response.json()
   return  "https://image.tmdb.org/t/p/w500/" + data['poster_path']

def recommend(movie):
  movie_index = movies[movies['title'] == movie].index[0]
  distances = similarity[movie_index]
  movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
  
  recommended_movies = []
  recommended_movies_posters = []
  for i in movies_list:
    movie_id = movies.iloc[i[0]].movie_id
    # fetch poster from API
    recommended_movies.append(movies.iloc[i[0]].title)
    recommended_movies_posters.append(fetch_poster(movie_id))
  return recommended_movies,recommended_movies_posters


movies_dict = pickle.load(open('movies.pkl','rb'))
movies = pd.DataFrame(movies_dict)
similarity = pickle.load(open('similarity.pkl','rb'))

st.title('🎥 Movie Recommender System')
st.subheader('Discover movies tailored to your taste!')
selected_movie_name = st.selectbox('Select a movie you like:', movies['title'].values)

with st.spinner('Fetching recommendations...'):
    names, posters = recommend(selected_movie_name)
st.subheader('Recommended Movies')

cols = st.columns(5) 
for idx, col in enumerate(cols):
    with col:
        st.image(posters[idx], use_column_width=True)
        st.markdown(f"<p style='text-align: center;'>{names[idx]}</p>", unsafe_allow_html=True)