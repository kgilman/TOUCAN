import numpy as np
import matplotlib.pyplot as plt
import time
from tsvd import *

def unfold(tensor,mode=0):
    return np.reshape(np.moveaxis(tensor,mode,0),(tensor.shape[mode],-1),order = 'F')

def CPcomp(lam,A,B,C):
    n1 = A.shape[0]
    n2 = B.shape[0]
    n3 = C.shape[0]

    X = np.zeros((n1,n2,n3))

    for i in range(0,n3):
        X[:,:,i] = (lam * A) @ np.diag(C[i,:]) @ B.T

    return X

def frobnorm(X):
    return np.sqrt(np.sum(X**2))

def RPTM(T,maxiter):
    n1 = T.shape[0]
    n2 = T.shape[1]
    n3 = T.shape[2]

    uinit = np.random.randn(n1)
    u1 = uinit / np.linalg.norm(uinit)
    uinit = np.random.randn(n2)
    u2 = uinit / np.linalg.norm(uinit)
    uinit = np.random.randn(n3)
    u3 = uinit / np.linalg.norm(uinit)

    for iter in range(0,maxiter):
        v1 = np.zeros(n1)
        v2 = np.zeros(n2)
        v3 = np.zeros(n3)

        for i in range(0,n3):
            v3[i] = u1.T @ T[:,:,i] @ u2
            v1 += u3[i] * T[:,:,i] @ u2
            v2 += u3[i] * T[:,:,i].T @ u1

        u10 = u1.copy()
        u1 = v1 / np.linalg.norm(v1)
        u20 = u2.copy()
        u2 = v2 / np.linalg.norm(v2)
        u30 = u3.copy()
        u3 = v3 / np.linalg.norm(v3)

        if(np.linalg.norm(u10 - u1) + np.linalg.norm(u20 - u2) + np.linalg.norm(u30 - u3) < 1e-7):
            break

        return u1,u2,u3

def TenProj(T,U1,U2,U3):
    if(len(U1.shape) < 2):
        U1 = np.expand_dims(U1,axis=1)
        U2 = np.expand_dims(U2, axis=1)
        U3 = np.expand_dims(U3, axis=1)

    n1,r1 = U1.shape
    n2,r2 = U2.shape
    n3,r3 = U3.shape
    M = np.zeros((r1,r2,r3))
    for i in range(0,r3):
        A = np.zeros((n1,n2))
        for j in range(0,n3):
            A += T[:,:,j] * (U3[j,i] * np.ones((n1,n2)))

        M[:,:,i] = U1.T @ A @ U2
    return M

def TenALS(TE,E,r,max_iter,tol,init='rptm',ninit = 10, fun = lambda Xhat: [0,0], verbose = False):

    if(init is 'rptm'):
        n1 = TE.shape[0]
        n2 = TE.shape[1]
        n3 = TE.shape[2]
        U01 = np.zeros((n1,r))
        U02 = np.zeros((n2,r))
        U03 = np.zeros((n3,r))
        S0 = np.zeros(r)

        for i in range(0,r):
            tU1 = np.zeros((n1,ninit))
            tU2 = np.zeros((n2,ninit))
            tU3 = np.zeros((n3,ninit))
            tS = np.zeros(ninit)

            for j in range(0,ninit):
                a,b,c = RPTM(TE - CPcomp(S0,U01,U02,U03), max_iter)
                a /= np.linalg.norm(a)
                b /= np.linalg.norm(b)
                c /= np.linalg.norm(c)

                tU1[:,j] = a
                tU2[:,j] = b
                tU3[:,j] = c
                tS[j] = TenProj(TE - CPcomp(S0,U01,U02,U03),a,b,c)

            I = np.argmax(tS)

            U01[:,i] = tU1[:,I] / np.linalg.norm(tU1[:,I])
            U02[:, i] = tU2[:, I] / np.linalg.norm(tU2[:, I])
            U03[:, i] = tU3[:, I] / np.linalg.norm(tU3[:, I])
            S0[i] = TenProj(TE - CPcomp(S0,U01,U02,U03),U01[:,i],U02[:,i],U03[:,i])
    else:

        ### SVD INITIALIZATION STAGE
        TE1 = unfold(TE,0)
        TE2 = unfold(TE,1)
        TE3 = unfold(TE,2)


        U01 = np.linalg.svd(TE1,full_matrices=False)[0][:,:r]
        U02 =  np.linalg.svd(TE2,full_matrices=False)[0][:,:r]
        U03 =  np.linalg.svd(TE3,full_matrices=False)[0][:,:r]

        S0 = np.ones(r)


    ### ALS STAGE
    V1 = U01.copy()
    V2 = U02.copy()
    V3 = U03.copy()

    n1 = V1.shape[0]
    n2 = V2.shape[0]
    n3 = V3.shape[0]

    S = S0.copy()

    normTE = 0
    for i in range(0, n3):
        normTE += np.linalg.norm(TE[:, :, i], 'fro') ** 2

    err_log = []
    times = []
    stats = np.zeros((max_iter + 1,3))
    Xhat = CPcomp(S,V1,V2,V3)
    cost,nrmse = fun(Xhat)
    stats[0,:] = [cost, nrmse, 0]
    for iter in range(0,max_iter):
        tstart = time.time()
        # V1_ = V1
        # V2_ = V2
        # V3_ = V3

        for q in range(0,r):
            S_ = S.copy()
            S_[q] = 0
            A = CPcomp(S_,V1,V2,V3) * E
            # v1 = V1[:,q]
            v2 = V2[:,q].copy()
            v3 = V3[:,q].copy()

            ### Initialize: Zero out the qth component of V's
            V1[:,q] = np.zeros(n1)
            V2[:,q] = np.zeros(n2)
            V3[:,q] = np.zeros(n3)
            den1 = np.zeros(n1)
            den2 = np.zeros(n2)
            # s = S[q]

            ### Update v1
            for i in range(0,n3):
                V1[:,q] += v3[i] * (TE[:,:,i] - A[:,:,i])@v2
                den1 += v3[i]**2 * E[:,:,i] @ (v2 * v2)
            v1 = V1[:,q] / den1
            v1 /= np.linalg.norm(v1)

            ### Update v2
            for i in range(0,n3):
                V2[:,q] += v3[i] * (TE[:,:,i] - A[:,:,i]).T @ v1
                den2 += v3[i]**2 * E[:,:,i].T @ (v1 * v1)
            v2 = V2[:,q] / den2
            v2 /= np.linalg.norm(v2)

            ### Update v3
            for i in range(0,n3):
                V3[i,q] = (v1.T @ (TE[:,:,i] - A[:,:,i]) @ v2) / ((v1 * v1).T @ E[:,:,i] @ (v2 * v2))
                if(V3[i,q] == np.nan):
                    print('NaN is als_tensor, denominator=0 \n')
                    break

            ### Error Diagnostics
            # if(np.nonzero(den1) != n1 or np.nonzero(den2) != n2):
            #     print('NaN is als_tensor, denominator = 0 \n')
            #     break

            if(np.linalg.norm(V1[:,q])==0 or np.linalg.norm(V2[:,q])==0 or np.linalg.norm(V3[:,q])==0):
                print('ERROR: estimate 0!')

            ### Store results
            V1[:,q] = v1.copy()
            V2[:,q] = v2.copy()
            S[q] = np.linalg.norm(V3[:,q])
            V3[:,q] /= np.linalg.norm(V3[:,q])

        ### Termination Diagnostics
        # err = TE - E * CPcomp(S,V1,V2,V3)
        # normErr = 0
        # for i in range(0,n3):
        #     normErr += np.linalg.norm(err[:,:,i],'fro')**2
        #
        # err_log.append(np.sqrt(normErr / normTE))
        # times.append(time.time() - tstart)
        # if(np.sqrt(normErr / normTE) < tol):
        #     break
        tElapsed = time.time() - tstart
        Xhat = CPcomp(S,V1,V2,V3)
        cost,nrmse = fun(Xhat)
        stats[iter + 1,:] = [cost,nrmse,tElapsed]

        if(nrmse < tol):
            stats = stats[:iter + 1,:]
            break

        if(verbose and iter % 10 ==0):
            print('Iter[{:d}]: Cost fxn: {:.3f}, NRMSE: {:.6f} '.format(iter,cost,nrmse))

    return Xhat,V1,V2,V3,stats


# def main():
#     r = 5
#     n1 = 50
#     n2 = 40
#     n3 = 30

#     # n1 = 100
#     # n2 = 100
#     # n3 = 100
#     np.random.seed(0)

#     A = np.random.randn(n1,r)
#     A /= np.linalg.norm(A,axis=0)
#     B = np.random.randn(n2,r)
#     B /= np.linalg.norm(B,axis=0)
#     C = np.random.randn(n3,r)
#     C /= np.linalg.norm(C,axis=0)

#     X = CPcomp(np.ones(r),A,B,C)

#     U,S,V = tsvd(Tensor(X),full=False)
#     s0 = np.diag(S.array()[:,:,0])

#     # for i in range(1,r):
#     #     S.array()[i,i,:] = S.array()[i,i,:] * (s0[i-1] / s0[i] * 0.999/(s0[i-1] / s0[i]))

#     s0 = np.diag(S.array()[:,:,0])
#     plt.plot(s0)
#     plt.show()
#     print(s0)
#     cond_num = s0[0] / s0[r-1]
#     print(cond_num)

#     X = (U * S * V.T()).array()

#     P = Tensor(np.random.randn(n1,r,n3))
#     Q = Tensor(np.random.randn(n2,r,n3))

#     X2 = (P * Q.T()).array()
#     U2, S2, V2 = tsvd(Tensor(X2), full=False)
#     s02 = np.diag(S2.array()[:, :, 0])
#     plt.plot(s02)
#     plt.show()
#     cond_num = s02[0] / s02[r - 1]
#     print(cond_num)

#     rho = 0.8
#     mask = np.random.rand(n1,n2,n3)
#     mask[mask > rho] = 1
#     mask[mask <= rho] = 0
#     mask = mask.astype(int)
#     M = mask
#     Y = M * X

#     max_iter = 100
#     fun = lambda Xhat: [0, frobnorm(Xhat - X) / frobnorm(X)]
#     Xhat,V1,V2,V3,stats = TenALS(Y,M,r,max_iter = max_iter,tol=1e-8,fun=fun,verbose=False)

#     nrmse = stats[:,1]

#     plt.semilogy(np.arange(0,len(nrmse)),nrmse)
#     plt.ylabel('Error')
#     plt.xlabel('Iteration')
#     plt.xticks(np.arange(0,len(nrmse),5))
#     plt.show()

# main()

