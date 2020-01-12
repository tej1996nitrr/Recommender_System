# Neural Networks for Collaborative Filtering
#implementing https://www.comp.nus.edu.sg/~xiangnan/papers/ncf.pdf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
dataset = pd.read_csv("Data/ml-100k/u.data",sep = '\t',names="user_id,item_id,rating,timestamp".split(","))
dataset.head()
len(dataset.user_id.unique()), len(dataset.item_id.unique())
'''We assign a unique number between (0, #users) to each user and do the same for movies.'''
dataset.user_id = dataset.user_id.astype('category').cat.codes.values
dataset.item_id = dataset.item_id.astype('category').cat.codes.values
dataset.head()