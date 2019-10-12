import pandas as pd
credit = pd.read_csv(r'F:\PyCharm\Recommender_System\Data\tmdb_5000_credits.csv')
credit.columns = ['id','title','cast','crew']
features = ['cast', 'crew', 'keywords', 'genres']
metadata = pd.read_csv(r'F:\PyCharm\Recommender_System\Data\tmdb_5000_movies.csv',low_memory=False)
metadata.columns

credit.columns
movie_data = metadata.merge(credit,on='id')
movie_data.columns

'''Preprocessing/Cleaning Data'''
def get_actors(x):
    c = eval(x)
    l = []
    for j in c:
        l.append(j['name'])
    return l
movie_data['actors']=movie_data['cast'].apply(get_actors)

def get_director(x):
    crew_list = eval(x)
    l=[]
    for i in crew_list:
        if i['department'] == 'Directing':
            l.append(i['name'])
    return l
movie_data['director']=movie_data['crew'].apply(get_director)

def get_keywords(x):
    c = eval(x)
    l = []
    for j in c:
        l.append(j['name'])
    return l
movie_data['keywords'] = movie_data['keywords'].apply(get_keywords)

def get_genre(x):
    c = eval(x)
    l = []
    for j in c:
        l.append(j['name'])
    return l
movie_data['genres'] = movie_data['genres'].apply(get_genre)

def clean_data(x):
    #removing all spaces and lowercasing each item of list
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        #Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''

features = ['actors', 'keywords', 'director', 'genres']

df2=movie_data
for feature in features:
    df2[feature] = df2[feature].apply(clean_data)

'''Creating metadata for feeding vectorizer'''

df2['soup'] = df2['keywords'] + df2['actors'] + df2['director'] + df2['genres']
def join_strings(x):
    s=""
    for i in x:
        s=s+i+" "
    return s
df2['text'] = df2['soup'].apply(join_strings)

'''Vectorizing'''

from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df2['text'])
count_matrix.shape

'''Computing the Cosine Similarity matrix based on the count_matrix'''
from sklearn.metrics.pairwise import cosine_similarity
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

'''Reseting index of DataFrame and construct reverse mapping for the 'get_recommendation' function'''
df2 = df2.reset_index()
indices = pd.Series(df2.index, index=df2['title_x'])

'''Function to get recommendations based on cosine similarity'''
def get_recommendations(title,cosine_sim = cosine_sim2):
    idx =indices[title]
    sim_score = list(enumerate(cosine_sim[idx]))
    sim_score = sorted(sim_score,key=lambda x:x[1], reverse=True)
    sim_score =sim_score[1:11]
    movie_indices = [i[0] for i in sim_score]
    return metadata['title'].iloc[movie_indices]

get_recommendations('The Dark Knight Rises', cosine_sim2)
get_recommendations('The Godfather',cosine_sim2)
get_recommendations('Mean Girls',cosine_sim2)
