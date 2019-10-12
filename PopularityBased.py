import pandas as pd
metadata = pd.read_csv(r'movies_metadata2.csv',low_memory=False)
metadata.head(3)
metadata.columns
#mean vote across the whole report
C = metadata['vote_average'].mean()

# Calculating the minimum number of votes required to be in the chart, m
m = metadata['vote_count'].quantile(0.90)

q_movies = metadata.copy().loc[metadata['vote_count'] >= m]
q_movies.shape

def weighted_rating(x,m=m,c=C):
    v=x['vote_count']
    R=x['vote_average']
    return (v/(v+m))*R +(m/(m+v))*C

q_movies['score'] = q_movies.apply(weighted_rating,axis=1)
q_movies =q_movies.sort_values('score',ascending=False)
q_movies[['title', 'vote_count', 'vote_average', 'score']].head(15)