import numpy as np
from tsvd import *
from stc import *
import matplotlib.pyplot as plt

t = 60
o = 50
d = 100

K = 3
U1 = np.linalg.svd(np.random.randn(o,t*d),full_matrices=False)[0][:,:K]
W1 = np.random.randn(K,t*d)
X1 = U1 @ W1
X1 = np.transpose(np.reshape(X1,(o,t,d)),[1,0,2])

U2 = np.linalg.svd(np.random.randn(t,o*d),full_matrices=False)[0][:,:K]
W2 = np.random.randn(K,o*d)
X2 = U2 @ W2
X2 = np.reshape(X2,(t,o,d))

U3 = np.linalg.svd(np.random.randn(o*t,d),full_matrices=False)[0][:,:K]
W3 = np.random.randn(K,d)
X3 = U3 @ W3
X3 = np.reshape(X3,(t,o,d))

L = 1/3 * (X1 + X2 + X3)

# r =  3
# n1 = 60
# n2 = 100
# n3 = 50
#
# np.random.seed(0)
#
# P = tsvd(Tensor(np.random.randn(n1, n1, n3)))[0]
# # P = Tensor(np.random.randn(n1,n1,n3))
# Utrue = Tensor(P.array()[:, :r, :])
#
# W = Tensor(np.random.randn(r, n2, n3))
# L = (Utrue * W).array()
#
# t = n1
# o = n2
# d = n3
# K = r

# U,S,V = tsvd(Tensor(L))
#
# plt.plot(np.diag(S.array()[:,:,0]))
# plt.show()

tube = False
rho = 0.5  # percentage of missing entries

if (tube is False):
    mask = np.random.rand(t, o, d)
    mask[mask > rho] = 1
    mask[mask <= rho] = 0
    mask = mask.astype(int)
else:
    mask = np.random.rand(t, o)
    mask[mask > rho] = 1
    mask[mask <= rho] = 0
    mask = mask.astype(int)
    mask = np.repeat(mask[:, :, np.newaxis], d, axis=2)

Lfrob = tfrobnorm_array(L)
Y = L * mask

numcycles = 1
outer = 5
r1 = 6
r2 = 6
r3 = 6
fun = lambda Lhat,idx: [0, tfrobnorm_array(Lhat[:,:,idx] - L[:,:,idx]) / tfrobnorm_array(L[:,:,idx])]
Lhat, stats, tElapsed = stc(Y,mask,r1,r2,r3,outer,numcycles,fun=fun,verbose=True)

