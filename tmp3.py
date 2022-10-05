import torch
import torch.nn as nn
import torch.distributions as tdist
import numpy as np
import torch.nn.functional as F

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

a = np.zeros(10)
a[3] = 1
a[7] = 1

idx = np.where(a == 1)[0]
idx2 = idx + 1
print(idx)
print(idx2)
