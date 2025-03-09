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

# preview the first 5 rows of X
print (X[0:5])

# create neural network to predict

# define neural network
model = nn.Sequential(
    nn.Linear(3,16),
    nn.ReLU(),
    nn.Linear(16,8),
    nn.ReLU(),
    nn.Linear(8,4),
    nn.ReLU(),
    nn.Linear(4,1)
)

# make predictions (without learning)
predicted_rent = model(X)

print (predicted_rent[0:5])