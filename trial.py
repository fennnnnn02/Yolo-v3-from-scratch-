import numpy as np
import torch 

g = np.arange(10)
print(g)

a,b = np.meshgrid(g,g)

a = torch.FloatTensor(a).view(-1,1)
b = torch.FloatTensor(b).view(-1,1)

ab = torch.cat((a,b),1).repeat(1,5).view(-1,2).unsqueeze(0)

print(ab)