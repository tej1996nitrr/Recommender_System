#%%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
print("hello")

# %%
movies = pd.read_csv('Data\ml-1m\movies.dat',sep='::',header= None,engine ='python',encoding = 'latin-1')
ratings = pd.read_csv(r'Data\ml-1m\ratings.dat',sep='::',header= None,engine ='python',encoding = 'latin-1')

users = pd.read_csv(r'Data\ml-1m\users.dat',sep='::',header= None,engine ='python',encoding = 'latin-1')

# Preparing the training set and the test set
training_set = pd.read_csv('Data/ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('Data/ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')


# %%
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

# Converting the data into an array with users in lines and movies in columns
def convert(data):
        #making a list of lists, where we will have 943 lists(no of users )  and 1682 elements(no of movies)

    new_data = []
    for id_users in range(1, nb_users + 1):
        #data[:,1] => taking movie ids
        # we want the movie ids acc to each user id =>id_movies = data[:,1][data[:,0]==id_users]
        # getting ratings of the movies=>id_ratings = data[:,2][data[:,0]==id_users]
        #col0=userid, col1=movieid,col2=rating
        #debug
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        #above is the ratings of the movie that user rated.however we also want the zeros whwere user hasnt rated
        #we create list of 1682 elements which will include zeros
        #movie id start at 1 python index starts as 0 hence id_movies-1
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Converting the ratings into binary ratings 1 (Liked) or 0 (Not Liked)
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1
test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1


# %%
# Creating the architecture of the Neural Network
#Contrastive  Divergence techinque
'''visible nodes=inputs=ratings of the movies'''
class RBM():
    def __init__(self, nv, nh):
        self.W = torch.randn(nv, nh) # was torch.randn(nh, nv)
        self.a = torch.randn(1, nh)
        self.b = torch.randn(1, nv)
   
    def sample_h(self, x):
        wx = torch.mm(x, self.W) # was torch.mm(x, self.W.t())    
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    
    def sample_v(self, y):
        wy = torch.mm(y, self.W.t()) # was torch.mm(y, self.W)  
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)

    def train(self, v0, vk, ph0, phk):
        self.W += torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)

nv = len(training_set[0])
nh = 100
batch_size = 100#we wont update weightsafter each observations but after 100 operations
rbm = RBM(nv, nh)

# %%
nb_epoch = 10
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(0, nb_users - batch_size, batch_size):
        vk = training_set[id_user:id_user+batch_size]#this will go in gibbs chain
        v0 = training_set[id_user:id_user+batch_size]#target,used for getting the loss
        ph0,_ = rbm.sample_h(v0)#probability that hidden node at start =1 givrn the real ratings that are already rated in the batch
        for k in range(10):
            _,hk = rbm.sample_h(vk)#vk=v0
            _,vk = rbm.sample_v(hk)#update vk 
            vk[v0<0] = v0[v0<0]#our ratings are either -1,or 0 or 1 so we retain the -1 ratings 
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
        s += 1.
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))


 # %%
test_loss = 0
s = 0.
for id_user in range(nb_users):
    v = training_set[id_user:id_user+1]
    vt = test_set[id_user:id_user+1]
    if len(vt[vt>=0]) > 0:
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
        s += 1.
print('test loss: '+str(test_loss/s))

# %%
