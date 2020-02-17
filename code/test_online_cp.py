from tsvd import *
from tecpsgd import *
import numpy as np
import time
import matplotlib.pyplot as plt

r =  5
n1 = 100
n2 = 100
n3 = 400
np.random.seed(0)

A=np.random.randn(n1, r)
B=np.random.randn(n2, r)
C=np.random.randn(n3, r)

#Create observed tensor that follows PARAFAC model
Tensor_Y_Noiseless = np.zeros((n1,n2,n3))
for k in range(0,n3):
    Tensor_Y_Noiseless[:,:,k]=A@np.diag(C[k,:])@B.T

L = Tensor(Tensor_Y_Noiseless)

# np.random.seed(0)
# P = tsvd(Tensor(np.random.randn(n1,n1,n3)))[0]
# Utrue = Tensor(P.array()[:,:r,:])
#
# W = Tensor(np.random.randn(r, n2, n3))
# L = Utrue * W

tube = False
rho = 0.5  # percentage of missing entries

if(tube is False):
    mask = np.random.rand(n1, n2, n3)
    mask[mask > rho] = 1
    mask[mask <= rho] = 0
    mask = mask.astype(int)
else:
    mask = np.random.rand(n1, n2)
    mask[mask > rho] = 1
    mask[mask <= rho] = 0
    mask = mask.astype(int)
    mask = np.repeat(mask[:, :, np.newaxis], n3, axis=2)

##############
Tensor_Y_Noiseless = L.array()
rank = r
OmegaTensor = mask
tensor_dims = [n1,n2,n3]
maxepochs = 1
tolcost = 1e-14
permute_on = False

options = {
    'maxepochs': maxepochs,
    'tolcost': tolcost,
    'lam': 0.001,
    'stepsize': 0.1,
    'mu': 0.05,
    'permute_on':  permute_on,
    'store_subinfo': True,
    'store_matrix': False,
    'verbose': True
}

Xinit = {
    'A': np.random.randn(tensor_dims[0], rank),
    'B': np.random.randn(tensor_dims[1], rank),
    'C': np.random.randn(tensor_dims[2], rank)
}

Xsol_TeCPSGD, info_TeCPSGD, sub_infos_TeCPSGD = TeCPSGD(Tensor_Y_Noiseless, OmegaTensor, None, tensor_dims, rank,
                                                          Xinit, options)

plt.semilogy(sub_infos_TeCPSGD['err_residual'][1:])
plt.show()
print("I'm done")