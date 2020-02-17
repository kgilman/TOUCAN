import numpy as np
import time
import matplotlib.pyplot as plt
from tsvd import *
from tecpsgd import *
from olstec import *

r =  5
n1 = 100
n2 = 100
n3 = 100

# r =  3
# n1 = 50
# n2 = 40
# n3 = 30

np.random.seed(0)

# P = tsvd(Tensor(np.random.randn(n1, n1, n3)))[0]
# # P = Tensor(np.random.randn(n1,n1,n3))
# Utrue = Tensor(P.array()[:, :r, :])

# W = Tensor(np.random.randn(r, n2, n3))
# L = Utrue * W

# tube = False
# rho = 0.8  # percentage of missing entries

# if (tube is False):
#     mask = np.random.rand(n1, n2, n3)
#     mask[mask > rho] = 1
#     mask[mask <= rho] = 0
#     mask = mask.astype(int)
# else:
#     mask = np.random.rand(n1, n2)
#     mask[mask > rho] = 1
#     mask[mask <= rho] = 0
#     mask = mask.astype(int)
#     mask = np.repeat(mask[:, :, np.newaxis], n3, axis=2)

# sig = 0

# L += Tensor(sig * np.random.randn(n1, n2, n3))

# Lfrob = tfrobnorm(L)


# A=np.linalg.svd(np.random.randn(n1, r),full_matrices=False)[0]
# B=np.linalg.svd(np.random.randn(n2, r),full_matrices=False)[0]
# C=np.linalg.svd(np.random.randn(n3, r),full_matrices=False)[0]

A=np.random.randn(n1, r)
B=np.random.randn(n2, r)
C=np.random.randn(n3, r)
#
# #Create observed tensor that follows PARAFAC model
Tensor_Y_Noiseless = np.zeros((n1,n2,n3))
for k in range(0,n3):
    Tensor_Y_Noiseless[:,:,k]=A@np.diag(C[k,:])@B.T
L = Tensor(Tensor_Y_Noiseless)
Lfrob = tfrobnorm(L)
print(L.shape())

# Y_m = np.reshape(Tensor_Y_Noiseless,(n1*n3,n2))
# U,S,V = np.linalg.svd(Y_m,full_matrices=False)
# plt.plot(S)
# plt.xticks(np.arange(0, min(n1,n1), step=10))
# plt.show()

# U,S,V = tsvd(L,full=False)
# plt.figure(figsize=(13,5),tight_layout=True)
# plt.plot(np.arange(0,min(n1,n2)),np.diag(S.array()[:,:,0]))
# plt.xticks(np.arange(0, min(n1,n2), step=5))
# plt.title('Tubal singular values')
# # plt.yticks(np.arange(0,np.max(S.array()),step=500))
# plt.show()

# print(np.diag(S.array()[:,:,0]))

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

# Y = Tensor(L.array() * mask)

# # TOUCAN

# fun = lambda X,k: [0, tfrobnorm_array(X.array().squeeze() - L.array()[:,k,:]) / tfrobnorm_array(L.array()[:,k,:])]
# Y_hat_toucan, stats_toucan, tElapsed_toucan = toucan(Y,mask,rank=r,tube=tube,cgtol=1e-8,outer=5,fun=fun,
#                                                      randomOrder=False,verbose=True)

# toucan_nrmse = tfrobnorm(Y_hat_toucan - L) / Lfrob
# print('TOUCAN Time: {:4f}'.format(tElapsed_toucan))
# print('TOUCAN NRMSE: {:4f}'.format(toucan_nrmse))

# nrmse_toucan = stats_toucan[:,1]
# cgiter_toucan = stats_toucan[:,2]
# times_toucan = stats_toucan[:,-1]

# plt.semilogy(nrmse_toucan)
# plt.title('TOUCAN: Recovered Tensor NRMSE')
# plt.xlabel('Iteration')
# plt.show()

# plt.scatter(np.arange(0,len(cgiter_toucan[1:])),cgiter_toucan[1:])
# plt.show()

# Tensor_Y_Noiseless = np.transpose(L.array(),[0,2,1])
rank = r
# OmegaTensor = np.transpose(mask,[0,2,1])
OmegaTensor = mask
# tensor_dims = [n1,n3,n2]
tensor_dims = [n1,n2,n3]
maxepochs = 2
tolcost = 1e-14
permute_on = False


Xinit = {
    'A': np.random.randn(tensor_dims[0], rank),
    'B': np.random.randn(tensor_dims[1], rank),
    'C': np.random.randn(tensor_dims[2], rank)
}

options = {
    'maxepochs': maxepochs,
    'tolcost': tolcost,
    'lam': 0.001,
    'stepsize': 0.5,
    'permute_on':  permute_on,
    'store_subinfo': True,
    'store_matrix': False,
    'verbose': True
}

Xsol_TeCPSGD, Y_hat_tecpsgd, info_TeCPSGD, sub_infos_TeCPSGD = TeCPSGD(Tensor_Y_Noiseless, OmegaTensor, None, tensor_dims, rank,
                                                          Xinit, options)
#
options = {
    'maxepochs': maxepochs,
    'tolcost': tolcost,
    'lam': 0.7,
    'mu': 0.1,
    'permute_on':  permute_on,
    'store_subinfo': True,
    'store_matrix': False,
    'verbose': True,
    'tw_flag': None,
    'tw_len': None
}

Xsol_olstec, Y_hat_olstec, info_olstecx, sub_infos_olstec = OLSTEC(Tensor_Y_Noiseless, OmegaTensor, None, tensor_dims, rank,
                                                          Xinit, options)


plt.semilogy(np.arange(0,len(sub_infos_TeCPSGD['err_residual'][1:])), sub_infos_TeCPSGD['err_residual'][1:],
             np.arange(0, len(sub_infos_olstec['err_residual'][1:])), sub_infos_olstec['err_residual'][1:])
plt.xlabel('Slice')
plt.legend(['TeCPSGD','OLSTEC'])
plt.show()

plt.semilogy(np.cumsum(sub_infos_TeCPSGD['times'][1:]),sub_infos_TeCPSGD['err_residual'][1:],
             np.cumsum(sub_infos_olstec['times'][1:]), sub_infos_olstec['err_residual'][1:])
plt.xlabel('Time (s)')
plt.legend(['TeCPSGD','OLSTEC'])
plt.show()
print("I'm done")

