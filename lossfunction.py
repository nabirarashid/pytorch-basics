import numpy as np
import pandas as pd

import torch
import torch.nn as nn

# define prediction and target tensors

# for predictions to track gradients & tensor to have the attribute requires_grad set to True
# usually for nn.Parameter tensors already have this attribute set to True
predictions = torch.tensor([-6.9229, -29.8163, -16.0748, -13.2427, -14.1096], dtype=torch.float, requires_grad=True)

y = torch.tensor([2550, 11500, 3000, 4500, 4795], dtype=torch.float)

# making instance of MSELoss function
loss = nn.MSELoss()
MSE = loss(predictions,y)

# show output with the gradient function!
print("MSE Loss:", MSE**0.5)
print(MSE.grad_fn)
