'''

Step 1: start with dataset with n_features independent variables
Step 2: create a grid composed of nodes, each having weight vector of n_feature elements
Step 3: Randomly initialise values of weight vectors to small numbers close to 0 (but not 0)
Step 4: Select 1 random observation point from the dataset
Step 5: Compute euclidan distances from this point to the different neurons in the network
Step 6: Select the neuron that has the minimum distance to this point. This neuron is called the winning node
Step 7: Update the weights of the winning node to move it closer to the point
Step 8: Using a Gaussian neighborhood funciton of mean the winning node, update the weights of the winning
        winnning node neighbors to move them closer to the point. The neighborhood radius is the sigma in
        the Gaussian function
Step 9: Repeat Steps 1 to 5 and udpate the weights after each observation (Reinforcement Learning) or after
        a batch of observations (Batch Learning) until the network converges to a point where the neighborhood
        stops decreasing

'''
