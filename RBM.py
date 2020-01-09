#%%
import numpy as np
import pandas as pd 
import torch.nn as nn
import torch.utils.data 
from torch.autograd import Variable
import torch.optim as optim

movies = pd.read_csv('Data\ml-1m\movies.dat',sep='::',header= None,engine ='python',encoding = 'latin-1')
ratings = pd.read_csv(r'Data\ml-1m\ratings.dat',sep='::',header= None,engine ='python',encoding = 'latin-1')

users = pd.read_csv(r'Data\ml-1m\users.dat',sep='::',header= None,engine ='python',encoding = 'latin-1')
# Preparing the training set and the test set
training_set = pd.read_csv('Data/ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('Data/ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

# %%
# Getting the number of users and movies
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

#converting data into array/matrix with users in rows and movies in cols
def convert(data):
    #making a list of lists, where we will have 943 lists(no of users )  and 1682 elements(no of movies)
    newdata=[]
    for id_users in range(1,nb_users+1):
        #data[:,1] => taking movie ids
        # we want the movie ids acc to each user id =>id_movies = data[:,1][data[:,0]==id_users]
        # getting ratings of the movies=>id_ratings = data[:,2][data[:,0]==id_users]
        #col0=userid, col1=movieid,col2=rating
        #debug
        id_movies = data[:,1][data[:,0]==id_users] #array of all the movie id that are rated
        id_ratings = data[:,2][data[:,0]==id_users] #array of all the ratings
        #above is the ratings of the movie that user rated.however we also want the zeros whwere user hasnt rated
        #we create list of 1682 elements which will include zeros
        ratings = np.zeros(nb_movies)
        #movie id start at 1 python index starts as 0 hence id_movies-1
        s=id_movies-1
        ratings[id_movies - 1] = id_ratings
        newdata.append(list(ratings))
    return newdata
training_set2 = convert(training_set)
test_set2 = convert(test_set)

training_tensor  = torch.FloatTensor(training_set2)
test_tensor = torch.FloatTensor(test_set2)


# Converting the ratings into binary ratings 1 (Liked) or 0 (Not Liked)
training_tensor[training_tensor == 0] = -1
training_tensor[training_tensor == 1] = 0
training_tensor[training_tensor == 2] = 0
training_tensor[training_tensor >= 3] = 1
test_tensor[test_tensor == 0] = -1
test_tensor[test_tensor == 1] = 0
test_tensor[test_tensor == 2] = 0
test_tensor[test_tensor >= 3] = 1


# %%
# Creating the architecture of the Neural Network
#Contrastive  Divergence techinque
'''visible nodes=inputs=ratings of the movies'''
class RBM():
    def __init__(self, nv, nh):
        #a is bias for p (h given v)
        #b is bias for p (v given h)
        self.W = torch.randn(nv, nh) # was torch.randn(nh, nv)   #100*1682
        self.a = torch.randn(1, nh) #1=> batch nh=> bias  #1*100
        self.b = torch.randn(1, nv) #1=> batch nv=> bias  #1*1682 
    def sample_h(self,x):
        wx = torch.mm(x, self.W) #100*1682 * 1682*100 = 100*100
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)#100*100
        return p_h_given_v, torch.bernoulli(p_h_given_v) #since we ae predicting if user will like a movie or not, it is bernouille
    def sample_v(self, y):
        '''we have this vector of probabilities
            p_v_given_h, and from this vector
            of probabilities we return some sampling
            of the visible nodes
            That is let's say that the if visible node
            has a probability of 0.25,
            then we take a random number
            between zero and one.
            If this number is below 0.25,
            then this visible node will get the value one.
            So that means that we predict
            that the movie corresponding to that visible node
            will get a like by the user,
            and if this random number is larger
            than 0.25, then this visible node
            will get the value zero.
            And, therefore, we predict that the movie
            corresponding to that visible node
            will not get a like by the user.'''
        #no of visible nodes =no of movies =1682
        wy = torch.mm(y, self.W.t()) #100*100 * 100*1682 = 100*1682
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation) #100*1682
        return p_v_given_h, torch.bernoulli(p_v_given_h)#This is vector of probabilities of the visible nodes.
        # each of the probability 

    def train(self, v0, vk, ph0, phk):
        ''' v0= input vector containing the ratings of all movies by one user
            vk= visible nodes obtained after k samplings or k contrastive divergence(k rounds from visible nodes to hidden nodes and vice versa)
            ph0= vector of probabilities that at the first iteration the hidden nodes equal to one given the values of v0
            phk= vector of probabilities that at the kth iteration the hidden nodes equal to one given the visile values of vk
'''
#refer to algo on pg 15 of pdf
        self.W += torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk) #100*1682 #!= 1682*100 * 100*100 - 1682*100 * 100*100 = 1682*100 
        self.b += torch.sum((v0 - vk), 0) #0 is added to keep dimension of tensor as two
        self.a += torch.sum((ph0 - phk), 0)



# %%
nv = len(training_set[0])
nh = 100
batch_size = 100#we wont update weightsafter each observations but after 100 operations
rbm = RBM(nv, nh)


# %%
nb_epoch=10
for epoch in range(1,nb_epoch+1):
    train_loss = 0
    s = 0.
    for id_user in range(0, nb_users-batch_size,100):
        vk = training_tensor[id_user:id_user+batch_size]#this will go in gibbs chain
        v0 = training_tensor[id_user:id_user+batch_size]#target,used for getting the loss
        ph0,_ = rbm.sample_h(v0)#probability that hidden node at start =1 givrn the real ratings that are already rated in the batch
        for k in range(10):
            _,hk = rbm.sample_h(vk)#vk=v0
            _,vk = rbm.sample_v(hk)#update vk 
            vk[v0<0] = v0[v0<0] #our ratings are either -1,or 0 or 1 so we retain the -1 ratings 
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
        s += 1.
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s)) 


# %%
