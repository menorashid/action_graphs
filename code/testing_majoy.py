import torch
import numpy as np

a = np.random.rand(100,1024,1024)
a = torch.Tensor(a).cuda()

b = np.random.rand(100,1024,1024)
b = torch.Tensor(b).cuda()

iter = 0
while True:
	c = torch.bmm(a,b)
	if iter%100==0:
		print iter,c.size()
	iter+=1

