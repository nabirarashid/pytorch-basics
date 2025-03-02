import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# prepare input tensor

# load pandas dataframe
apartments_df = pd.read_csv("streeteasy.csv")

# convert pandas dataframe to numpy array
apartments_numpy = apartments_df[["size_sqft", "bedrooms", "building_age_yrs"]].values

# convert numpy array to torch tensor
X = torch.tensor(apartments_numpy, dtype=torch.float32)

# define class for neural network
class OneHidden(nn.Module):
    # initialize the class
    def __init__(self,numHiddenNodes):
        super(OneHidden, self).__init__()
        # initialize the layers
        # 3 input features, numHiddenNodes output features
        self.layer1 = nn.Linear(3, numHiddenNodes)
        # numHiddenNodes input features, 1 output feature
        self.layer2 = nn.Linear(numHiddenNodes, 1)
        # activation function
        self.relu = nn.ReLU()

    # define the forward pass
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

# initialize model w 10 hidden nodes
model = OneHidden(10)

# run foward pass with input tensor X
predicted_rent = model(X)

print (predicted_rent[0:5])