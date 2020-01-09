'''Simple recommenders: offer generalized recommendations to every user, based on movie popularity and/or genre.
The basic idea behind this system is that movies that are more popular and critically acclaimed will
 have a higher probability of being liked by the average audience. IMDB Top 250 is an example of this system.


Content-based recommenders: suggest similar items based on a particular item. This system uses item metadata,
 such as genre, director, description, actors, etc. for movies, to make these recommendations.
 The general idea behind these
recommender systems is that if a person liked a particular item, he or she will also like an item that is similar to it.

Collaborative filtering engines: these systems try to predict the rating or preference that a user would
 give an item-based on past ratings
and preferences of other users. Collaborative filters do not require item metadata like
its content-based counterparts.

One of the metrics is rating. It however doesn't take
the popularity of a movie into consideration.
Therefore, a movie with a rating of 9 from 10 voters will be considered 'better' than a movie with a rating
of 8.9 from 10,000 voters.
his metric will also tend to favor movies
with smaller number of voters with skewed
 and/or extremely high ratings. As the number
of voters increase, the rating of a movie regularizes and approaches towards a value that
is reflective of the movie's quality. It is more difficult to discern the quality
of a movie with extremely few voters.

A weighted rating that takes into account the average rating and the number of votes it has garnered is used.
Such a system will make sure that a movie with a 9 rating
from 100,000 voters gets a (far) higher score than a YouTube Web Series with the same rating but a few hundred voters.

Weighted Rating (WR) = (v/v+m).R+(m/v+m).C
v is the number of votes for the movie
m is the minimum votes required to be listed in the chart There is no right value for m.
R is the average rating of the movie
C is the mean vote across the whole report

Then we calculate the number of votes, m, received by a movie in the 90th percentile.

Content-based Recsys (plot):

compute the word vectors of each overview or document,
compute Term Frequency-Inverse Document Frequency (TF-IDF) vectors for each document. This will give you a matrix
where each column represents a word in the
overview vocabulary (all the words that appear in at least one document) and each column represents a movie, as before.
the TF-IDF score is the frequency of a word occurring in a document, down-weighted by the number of documents in which it occurs.

what is term frequency , it is the relative frequency of a word in a document and is given as (term instances/total instances). Inverse Document Frequency is the relative count of documents containing the term is given as log(number of documents/documents with term)
The overall importance of each word to the documents in which they appear is equal to TF * IDF

Firstly, for this, we need a reverse mapping of movie titles and DataFrame indices.
In other words, we need a mechanism to identify the index of a movie in the metadata , given its title

Content based Recsys(Crew, Keyword,Genre,director)
Same as above, except we need a few preprocessing steps.
One important difference is that we use the CountVectorizer() instead of TF-IDF.
This is because  we do not want to down-weight the presence of an actor/director
if he or she has acted or directed in relatively more movies. It doesn't make much intuitive sense.

 '''