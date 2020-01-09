#%%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.optim  as optim
from torch.autograd import Variable

# %%
movies = pd.read_csv('Data\ml-1m\movies.dat',sep='::',header= None,engine ='python',encoding = 'latin-1')
#columns = (movieid,movie name,theme/genre)
ratings = pd.read_csv(r'Data\ml-1m\ratings.dat',sep='::',header= None,engine ='python',encoding = 'latin-1')
#columns = (user_id,movie_id,ratings,timestamp)
users = pd.read_csv(r'Data\ml-1m\users.dat',sep='::',header= None,engine ='python',encoding = 'latin-1')
#columns = (userid, gender,age,user_job,zip_Code)
# Preparing the training set and the test set
#u1 is one of the train test split
#all u1 to u5 are for cross validation
#80-20% train test split
training_set = pd.read_csv('Data/ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('Data/ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')


# %%
#max user_id can be in test set or in train set so we take max of both's max
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

# Converting the data into an array with users in lines/rows and movies in columns
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
# %%
class StackedAutoEnc(nn.Module):
    def __init__(self):
        super(StackedAutoEnc,self).__init__() #to get all the inherited methods and classes of the parent class
        self.fc1 = nn.Linear(nb_movies,20) #20 nodes in 1st hidden layer
        self.fc2 = nn.Linear(20,10)
        self.fc3 = nn.Linear(10,20)
        self.fc4 = nn.Linear(20,nb_movies)
        self.activation = nn.Sigmoid()
    def forward(self,x): # will return the output as vector of predicted ratings that we will compare to real ratings
        x= self.activation(self.fc1(x)) #will return first encoded vector
        x= self.activation(self.fc2(x))
        x= self.activation(self.fc3(x)) #decoding
        x = self.fc4(x) #no activation required in final part of decoding
        return x 
sae =StackedAutoEnc()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(),lr=0.01,weight_decay=0.5)

# %%
#training
nb_epoch = 100
for epoch in range(1,nb_epoch+1):
    train_loss = 0 
    s = 0. #to compute rmse in the end
    for id_user in range(nb_users):
        input = Variable(training_set[id_user]).unsqueeze(0) #all ratings of all movies given by id_user #pytorch dont accept single vector of one dimension so weadd an additional dimension which will correspond to a batch
        target = input.clone()
        #to optimize memory(only considering observations if user rated atleast one movie)
        if torch.sum(target.data>0) >0: #target.data = all ratings
            output  = sae(input)
            target.require_grad = False # we dont compute gradients for targets
            output[target==0]=0#these values will not count in computations of the error
            loss = criterion(output,target)
            mean_corrector = nb_movies/float(torch.sum(target.data>0)+1e-10) #average of the error of the movies that were rated,non zero ratings
            loss.backward()# decides whether weights will increase or decrease
            train_loss +=np.sqrt(loss.data*mean_corrector )  
            s+=1.
            optimizer.step()   # decides amount of weights to be increased or decrease
    print('epoch'+str(epoch)+' loss '+str(train_loss/s))      



    # %%
