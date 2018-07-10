# import all the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import dataset
dataset = pd.read_csv('Credit_Card_applications.csv')

# it's going to segmentate customers that potentially cheated
# customers are going to be inputs mapped into an output space
# each neuron is initialized as a vector of weights which is the same size as
# the vector of customers (15 elements)
# for each customer the outpue will be the neuron that is closest to the neuron
# this neuron is the winning neuron or BMU (best matching unit)
# update the weights of the nearby nodes and repeat them many times
# it will reduce the dimension little by little
# frauds are the outlier neurons in the SOMs
# thus we need an MID (mean inter-neuron distance)

# now we separate the dataset into the data and the results
# y is not going to be considered in the training
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# feature scaling REMEMBER
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
X = sc.fit_transform(X)

# now we train the SOM
# we can implement from scratch but we can also use the ones made by other developers
# we use MiniSOM for this
# sigma is the radius of the environment
# x and y defines the grid
# input_len is the number of inputs in the attributes
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)

# initialize random weights
som.random_weights_init(X)

# then we train our data
som.train_random(data = X, num_iteration = 100)

# now we need to visualize the data
# for each node we need to find the MID of a specific winning node
# the higher the MID, the more the winning node is an outlier
# outlying neuron outside of general rules
from pylab import bone, pcolor, colorbar, plot, show

bone()

# put nodes into the map
# information about the MID about the neurons
# diff colors to represent diff MID
# distance_map returns all MID in a matrix of the winning nodes
# we need to find the transpose that's why have .T
pcolor(som.distance_map().T)

# add legend
colorbar()

# frauds are identified by the high MID values
# next we need to inverse map it and mark it in the diagram
# customers who cheated but got approved is more significant
# need to mark corresponding to whether they got accepted or not
# add red circles not approved but green one got approval
markers = ['o', 's']
colors = ['r', 'g']

# we're going to loop through the customers and mark the SOMs
# i is going to be the different values of all the indexes of the customer database
# x is going to be the different vectors of the customers (all the attributes inside)
for i, x in enumerate(X):
    # get winning node for customer
    w = som.winner(x)
    # place color marker on the map to see whether customer got approval or not
    # specify coordinates (inside the centre of the square)
    # 0.5 to put middle of the square
    # take the approved and unapproved from y
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]], # y[i] corresponds to the y in the i th index
         markeredgecolor = colors[y[i]], # only color the edges
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
    
show()

# find the frauds!
# get the mappings in minisom and find the outliers
# concatenate the list
mappings = som.win_map(X) 

# find the ones with high MID (the coordinates)
# axis determines the concatenation dimension (vert or hor)
# we want to put along vertical axis
frauds = np.concatenate((mappings[(7, 3)], mappings[(4, 8)]), axis = 0)

# inverse the scale now
frauds = sc.inverse_transform(frauds)




