import pandas as pd

metadata = pd.read_csv(r'F:\PyCharm\Recommender_System\Data\tmdb_5000_movies.csv',low_memory=False)
metadata['overview'].head(5)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf =TfidfVectorizer(stop_words='english')

'''Plot description based Recommender'''
metadata['overview']=metadata['overview'].fillna('')
tfidf_matrix = tfidf.fit_transform(metadata['overview'])
tfidf_matrix.shape
from sklearn.metrics.pairwise import linear_kernel
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

#Construct a reverse map of indices and movie titles
indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()


def get_recommendations(title,cosine_sim = cosine_sim):
    idx =indices[title]
    sim_score = list(enumerate(cosine_sim[idx]))
    sim_score = sorted(sim_score,key=lambda x:x[1], reverse=True)
    sim_score =sim_score[1:11]
    movie_indices = [i[0] for i in sim_score]
    return metadata['title'].iloc[movie_indices]
get_recommendations('The Dark Knight Rises')





