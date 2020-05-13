import numpy as np
import time
from numpy import ndarray

class myarray(ndarray):
    @property
    def H(self):
        return self.conj().T

class Tensor:
    def __init__(self, array):
        ## input is n1 x n2 x n3 multidimensional array ###
        self.__n1 = array.shape[0]
        self.__n2 = array.shape[1]
        self.__n3 = array.shape[2]
        self.__array = array

    def T(self):  # transpose method
        At = np.zeros((self.__n2, self.__n1, self.__n3))
        At[:, :, 0] = self.__array[:, :, 0].T
        for i in range(1, self.__n3): At[:, :, self.__n3 - i] = self.__array[:, :, i].T

        return Tensor(At)
    def shape(self):    #return dimensions of tensors
        return self.__n1, self.__n2, self.__n3

    def array(self):    #return 4d numpy array of tensor
        return self.__array

    def __add__(self, B):
        return Tensor(self.__array + B.__array)

    def __sub__(self, B):
        return Tensor(self.__array - B.__array)

    def __mul__(self, B):  # tensor-tensor product method

        assert self.__n2 == B.__n1, "Dimensions of tensors must match"

        n3 = self.__n3

        Abar = np.fft.fft(self.__array, axis=2)
        Bbar = np.fft.fft(B.__array, axis=2)
        Cbar = np.zeros((self.__n1, B.__n2, n3), dtype=complex)

        for i in range(0, int(np.ceil((n3 + 1) / 2))): Cbar[:, :, i] = Abar[:, :, i] @ Bbar[:, :, i]

        for i in range(int(np.ceil((n3 + 1) / 2)), n3): Cbar[:, :, i] = np.conj(Cbar[:, :, n3 - i])

        C = np.real(np.fft.ifft(Cbar, axis=2))

        return Tensor(C)


def tfrobnorm(A):
    #
    # Computes \sum_{i,j,k} A[i,j,k]^2
    #
    # sum = 0
    # for i in range(0, A._Tensor__n3): sum += np.linalg.norm(A._Tensor__array[:, :, i], 'fro') ** 2
    # return np.sqrt(sum)
    return np.sqrt(np.sum(A.array()**2))


def tfrobnorm_array(A):
    #
    # Computes \sum_{i,j,k} A[i,j,k]^2
    #
    return np.sqrt(np.sum(A**2))


def tsvd(A, full=True):
    #
    # Compute the tensor-SVD from "Tensor Robust Principal Component Analysis with a New Tensor Nuclear Norm," avail: https://ieeexplore.ieee.org/abstract/document/8606166
    #
    # Input
    #   Object of class type Tensor, size n1 x n2 x n3
    #
    # Output
    #   Orthonormal U tensor of size n1 x n1 x n3
    #   F-diagonal S tensor of tubular singular values n1 x n2 x n3
    #   Orthonoraml V.T tensor of size n2 x n2 x n3

    n1, n2, n3 = A.array().shape
    Abar = np.fft.fft(A.array(), axis=2)

    K = min(n1, n2)
    if (full):
        Ubar = np.zeros((n1, n1, n3), dtype=complex)
        Sbar = np.zeros((min(n1, n2), n3), dtype=complex)
        Vbar = np.zeros((n2, n2, n3), dtype=complex)
    else:
        Ubar = np.zeros((n1, K, n3), dtype=complex)
        Sbar = np.zeros((K, n3), dtype=complex)
        Vbar = np.zeros((K, n2, n3), dtype=complex)

    for i in range(0, int(np.ceil((n3 + 1) / 2))):
        U, S, V = np.linalg.svd(Abar[:, :, i], full_matrices=full)
        Ubar[:, :, i] = U
        Sbar[:, i] = S
        Vbar[:, :, i] = V

    for i in range(int(np.ceil((n3 + 1) / 2)), n3):
        Ubar[:, :, i] = np.conj(Ubar[:, :, n3 - i])
        Sbar[:, i] = np.conj(Sbar[:, n3 - i])
        Vbar[:, :, i] = np.conj(Vbar[:, :, n3 - i])

    tU = Tensor(np.real(np.fft.ifft(Ubar, axis=2)))
    tV = Tensor(np.real(np.fft.ifft(Vbar, axis=2))).T()

    S = np.real(np.fft.ifft(Sbar, axis=1))
    if (full):
        tS = np.zeros((n1, n2, n3))
        for i in range(0, K): tS[i, i, :] = S[i, :]
        tS = Tensor(tS)
    else:
        tS = np.zeros((K, K, n3))
        for i in range(0, K): tS[i, i, :] = S[i, :]
        tS = Tensor(tS)

    return tU, tS, tV


def teye(n, n3):
    #
    # Function that returns Identity Tensor of size n x n x n3
    #
    I = np.expand_dims(np.eye(n), axis=2)
    I2 = np.zeros((n, n, n3 - 1))
    return Tensor(np.concatenate((I, I2), axis=2))


def tpinv(A):
    n1, n2, n3 = A.shape()
    Abar = np.fft.fft(A.array(),axis=2)
    Apinv_bar = np.zeros((n2,n1,n3),dtype=complex)

    for i in range(0, int(np.ceil((n3 + 1) / 2))):
        Apinv_bar[:,:,i] = np.linalg.pinv(Abar[:,:,i])
    for i in range(int(np.ceil((n3 + 1) / 2)), n3):
        Apinv_bar[:, :, i] = np.conj(Apinv_bar[:, :, n3 - i])


    return Tensor(np.fft.ifft(Apinv_bar,axis=2))

def normalizeTensorVec(v):
    v_F = np.fft.fft(v._Tensor__array, axis=2)
    vnorms_F = np.expand_dims(np.linalg.norm(v_F, axis=0), axis=0)
    vnormal_F = v_F / vnorms_F
    return vnormal_F, vnorms_F


def orthoTest(U):
    test = U.T() * U
    K = test._Tensor__n1
    n3 = test._Tensor__n3
    assert tfrobnorm(test - teye(K, n3)) < 1e-10

def tnn(A):
    #
    # Compute the tensor nuclear norm described in "Tensor Robust Principal Component Analysis with a New Tensor Nuclear Norm," avail: https://ieeexplore.ieee.org/abstract/document/8606166
    #
    U, S, V = tsvd(A, False)
    S = S._Tensor__array[:, :, 0]
    return np.sum(np.diag(S))

def tSVST(X, beta, full=False):
    #
    # Input: multidimensional array of size n1 x n2 x n3
    #
    # Output: multidimensional array of size n1 x n2 x n3
    #
    n1, n2, n3 = X.shape
    Xbar = np.fft.fft(X, axis=2)

    Wbar = np.zeros((n1, n2, n3), dtype=complex)

    for i in range(0, int(np.ceil((n3 + 1) / 2))):
        U, S, V = np.linalg.svd(Xbar[:, :, i], full_matrices=full)
        Wbar[:, :, i] = U @ np.diag(np.maximum(S - beta, 0.0)) @ V

    for i in range(int(np.ceil((n3 + 1) / 2)), n3):
        Wbar[:, :, i] = np.conj(Wbar[:, :, n3 - i])

    return np.real(np.fft.ifft(Wbar, axis=2))


def lrtc(Y, Mask, niter=400, min_iter = 10, rho = 1.1, mu = 1e-4, mu_max = 1e10, tol=1e-8, it_tol = 1e-4, fun=lambda X: [0, 0], verbose=False):
    #
    # Description: Iterative singular value soft thresholding algorithm for tensors
    #
    # Inputs
    #   Y: Tensor object of size n1 x n2 x n3
    #   mask: multidimensional binary array with 1's indicating observed entries
    #   beta: regularizatio parameter
    #   niter: number of ISTA iterations to perform
    #   fun: will evaluate cost function at current iterate X (but default is function -> 0)
    #
    # Outputs
    #   X: Tensor object object of size n1 x n2 x n3 with completed entries
    #   cost_ista: Cost function array of size niter + 1
    #
    Y = Y._Tensor__array
    X = Y.copy()
    X0 = X.copy()
    LAM = np.zeros(X.shape)
    stats = np.zeros((niter + 1,3))

    cost,nrmse = fun(Tensor(X))
    stats[0,:] = [cost, nrmse, 0]
    nrmse_0 = nrmse.copy()
    for k in range(0, niter):
        start = time.time()
        X = tSVST((Mask < 1).astype(float)*X + Mask*(Y - LAM / mu), 1 / mu)
        LAM = LAM + mu * (X - Y)

        mu = min(rho * mu, mu_max)

        tElapsed = time.time() - start

        cost, nrmse = fun(Tensor(X))
        stats[k + 1, :] = [cost, nrmse, tElapsed]
        it_diff = np.linalg.norm(X - X0)
        X0 = X.copy()

        if(nrmse < tol or (abs(nrmse - nrmse_0) < it_tol and k > min_iter)):
            stats = stats[:k+1,:]
            break

        nrmse_0 = nrmse.copy()

        if (verbose and k%10 == 0):
            print('Iter[{:d}]: Cost fxn: {:.3f}, NRMSE: {:.6f} '.format(k, cost, nrmse))
    
    tElapsed = np.sum(stats[:,2])
    return Tensor(X), stats, tElapsed

def tctf(Y,mask,rank,niter=100,min_iter = 10,tol=1e-8,it_tol = 1e-4,fun=lambda U,V: [0, 0],verbose=False):

    n1,n2,n3 = Y.shape()
    r = rank

    # X = Tensor(np.random.randn(n1, r, n3))
    # Z = Tensor(np.random.randn(r, n2, n3))

    U,S,V = tsvd(Y,full=False)
    X = Tensor(U.array()[:,:rank,:])
    Z = Tensor((S*V.T()).array()[:rank,:,:])

    stats = np.zeros((niter + 1, 3))

    # 0th iteration
    cost,nrmse = fun(X,Z)
    stats[0,:] = [cost,nrmse,0]
    C0 = X * Z
    nrmse_0 = nrmse.copy()
    for iter in range(0, niter):

        tStart = time.time()
        # C update
        C = X * Z + Tensor(np.multiply(mask, (Y - X * Z).array()))

        # Fourier Transforms
        Chat = np.fft.fft(C._Tensor__array, axis=2).view(myarray)
        Xhat = np.fft.fft(X._Tensor__array, axis=2).view(myarray)
        Zhat = np.fft.fft(Z._Tensor__array, axis=2).view(myarray)

        # X and Z updates in Fourier Domain
        # for i in range(0, n3):
        for i in range(0, int(np.ceil((n3 + 1) / 2))):
            Ci = Chat[:, :, i]
            Zi = Zhat[:, :, i]
            ZiH = Zi.H

            # X update
            Xi = Ci @ ZiH @ np.linalg.pinv(Zi @ ZiH)
            Xhat[:, :, i] = Xi

            # Z update
            XiH = Xi.H
            Zhat[:, :, i] = np.linalg.pinv(XiH @ Xi) @ XiH @ Ci

        for i in range(int(np.ceil((n3 + 1) / 2)), n3):
            Xhat[:, :, i] = np.conj(Xhat[:, :, n3 - i])
            Zhat[:, :, i] = np.conj(Zhat[:, :, n3 - i])

        # Inverse Fourier Transforms
        X = Tensor(np.real(np.fft.ifft(Xhat, axis=2)))
        Z = Tensor(np.real(np.fft.ifft(Zhat, axis=2)))

        tElapsed = time.time() - tStart

        cost, nrmse = fun(X,Z)
        stats[iter + 1,:] = [cost,nrmse,tElapsed]

        if(nrmse < tol or (abs(nrmse - nrmse_0) < it_tol and iter > min_iter)):
            stats = stats[:iter+1,:]
            break

        nrmse_0 = nrmse.copy()

        if(verbose):
            if(iter % 10 == 0):
                print('Iter[{:d}]: NRMSE: {:.6f} '.format(iter+1, nrmse))

        
    tElapsed = np.sum(stats[:,-1])
    return X,Z,stats,tElapsed


def lr_flatten(X):
    n1,n2,n3 = X.shape
    B = np.transpose(X, [2, 0, 1])
    X_m = np.reshape(B,(n1*n3,n2))

    return X_m


def lrmc(Y, Mask, niter=400, rho=1.1, mu=1e-4, mu_max=1e10, tol=1e-8, fun=lambda X: [0, 0], verbose=False):
    X = Y.copy()
    X0 = X.copy()
    LAM = np.zeros(X.shape)

    stats = np.zeros((niter + 1, 3))
    stats[0, :] = np.append(fun(X), 0)

    for k in range(0, niter):
        start = time.time()
        X = SVST((Mask < 1).astype(float) * X + Mask * (Y - LAM / mu), 1 / mu)
        LAM = LAM + mu * (X - Y)

        mu = min(rho * mu, mu_max)

        end = time.time()
        tElapsed = end - start

        cost, nrmse = fun(X)
        stats[k + 1, :] = [cost, nrmse, tElapsed]

        if(nrmse <tol):
            stats = stats[:k+1,:]
            break

        if (verbose and k % 10 == 0):
            print('Iter[{:d}]: Cost fxn: {:.3f}, NRMSE: {:.3f} '.format(k,cost, nrmse))

    tElapsed = np.sum(stats[:, -1])
    stats = stats[1:, :]
    return X, stats, tElapsed


def SVST(X, beta, full=False):
    U, S, V = np.linalg.svd(X, full_matrices=full)
    return U @ np.diag(np.maximum(S - beta, 0)) @ V


def trpca(T, mask, beta, niter, rho=1.1, mu=1e-4, fun=lambda L, S, Y, mu: [0, 0, 0], verbose=False):
    mu_max = 1e10
    eps = 1e-8

    X = T.array().copy()
    L0 = np.zeros(X.shape)
    S0 = np.zeros(X.shape)
    Y = np.zeros(X.shape)

    stats = np.zeros((niter + 1, 4))

    cost,lerr,serr = fun(L0,S0,Y,mu)

    stats[0,:] = [cost,lerr,serr,0]

    for iter in range(0, niter):
        tStart = time.time()

        L = tSVST(X - S0 - 1 / mu * Y, 1 / mu)
        S_p = soft(mask*(X - L - 1 / mu * Y), beta / mu)
        S_pc = (mask < 1).astype(float)*(X - L - 1/mu * Y)
        S = S_p + S_pc
        Y = Y + mu * (L + S - X)
        tElapsed = time.time() - tStart

        cost, lerr, serr = fun(L, S, Y, mu)

        mu = np.minimum(rho * mu, mu_max)

        stats[iter + 1, :] = [cost, lerr, serr, tElapsed]

        L0 = L.copy()
        S0 = S.copy()

        if (verbose):
            if(iter % 10 == 0):
                print('Iter {:.0f} [{:.3f}]: Cost {:6f}: L err: {:.6f}, S err {:.6f} '.format(iter, tElapsed*10, cost,
                                                                                              lerr,serr))

    return L0, S0, stats


def rpca(X, mask, beta, niter, rho=1.1, mu=1e-4, fun=lambda L, S, Y: [0, 0, 0], verbose=False):
    mu_max = 1e10
    eps = 1e-8

    L0 = np.zeros(X.shape)
    S0 = np.zeros(X.shape)
    Y = np.zeros(X.shape)

    stats = np.zeros((niter + 1, 4))

    cost,lerr,serr = fun(L0,S0,Y)

    stats[0,:] = [cost,lerr,serr,0]

    for iter in range(0, niter):
        tStart = time.time()
        L = SVST(X - S0 - 1 / mu * Y, 1 / mu)
        S_p = soft(mask*(X - L - 1 / mu * Y), beta / mu)
        S_pc = (mask < 1).astype(float)*(X - L - 1/mu * Y)
        S = S_p + S_pc
        Y = Y + mu * (L + S - X)
        tElapsed = time.time() - tStart

        cost, lerr, serr = fun(L, S, Y)

        mu = np.minimum(rho * mu, mu_max)

        stats[iter + 1, :] = [cost, lerr, serr, tElapsed]

        if (np.max(abs(L - L0)) < eps and np.max(abs(S - S0)) < eps and np.max(abs(L + S - X)) < eps):
            stats = stats[1:iter+1,:]
            break

        L0 = L.copy()
        S0 = S.copy()

        if (verbose):
            if(iter % 10 == 0):
                 print('Iter {:.0f} [{:.3f}]: Cost {:6f}: L err: {:.6f}, S err {:.6f} '.format(iter, tElapsed, cost,lerr,serr))

    return L0, S0, stats


def soft(X, beta):
    return np.sign(X) * np.maximum(np.abs(X) - beta, 0.0)


def nn(X):
    U, S, V = np.linalg.svd(X, False)
    return np.sum(S)

def grouse_stream(v, xIdx, U, step):

    ### Main GROUSE update


    U_Omega = U[xIdx,:]
    v_Omega = v[xIdx]
    w_hat = np.linalg.pinv(U_Omega)@v_Omega

    r = v_Omega - U_Omega @ w_hat

    rnorm = np.linalg.norm(r)
    wnorm = np.linalg.norm(w_hat)
    sigma = rnorm * np.linalg.norm(w_hat)

    if(step is None):
        t = np.arctan(rnorm / wnorm)
    else:
        t = step * sigma

    alpha = (np.cos(t) - 1) / wnorm**2
    beta = np.sin(t) / sigma
    Ustep = U @ (alpha * w_hat)
    Ustep[xIdx] = Ustep[xIdx] + beta * r
    Uhat = U + np.outer(Ustep, w_hat)


    return Uhat, w_hat

def grouse(Y,mask,rank,outer,mode,tol=1e-9,step=None,fun=lambda Lhat, idx: [0,0],randomOrder=False,verbose=False):

    n1,n2 = Y.shape

    ### Initialize U
    K = rank
    U = np.linalg.svd(np.random.randn(n1, n2),full_matrices=False)[0]
    U = U[:, :K]

    Lhat = np.zeros((n1, n2))
    What = np.zeros((rank,n2))

    stats = np.zeros((outer * n2 + 1, 3))
    # cost,nrmse = fun(Lhat)
    cost = 0
    nrmse = 1
    stats[0,:] = [cost,nrmse, 0]

    iter = 1
    for outer in range(0, outer):
        if (randomOrder):
            frame_order = np.random.permutation(n2)
        else:
            frame_order = np.arange(0,n2)

        for inner in range(0, n2):
            frame_idx = frame_order[inner]
            Yvec = Y[:, frame_idx]

            idx = np.where(mask[:, inner] > 0)[0]
            tStart = time.time()
            U, w = grouse_stream(Yvec, idx, U,step)
            tEnd = time.time()

            tElapsed = tEnd - tStart

            rec = U @ w

            if(mode == "online"):
                Lhat[:,inner] = rec
                cost, nrmse = fun(rec,frame_idx)
            else:
                # Lhat[:,inner] = rec
                What[:,frame_idx] = w
                Lhat = U @ What
                cost, nrmse = fun(Lhat)
                if(nrmse < tol):
                    stats = stats[:iter,:]
                    break

            stats[iter, :] = [cost, nrmse, tElapsed]
            iter += 1

            if(verbose):
                print('Outer[{:d}], Inner[{:d}]: NRMSE: {:.3f} '.format(outer, inner, nrmse))

    tElapsed = np.sum(stats[:,2])
    return Lhat, stats, tElapsed

def toucan_stream_tube(Yvec,idx,U,step=None):
    #
    # Description: performs one step of TOUCAN in the regime of missing TUBES to estimate n1 x K x n3 orthonormal
    # tensor U from input tensor column of size n1 x 1 x n3
    #
    # Inputs:
    #   Yvec: n1 x 1 x n3 tensor column (of type Tensor)
    #   idx: list of indices where Yvec is observed on the first dimension
    #   U: n1 x K x n3 orthonormal tensor (of type Tensor) initial estimate
    #   step: step size (real-valued constant)
    #
    # Outputs:
    #   U: updated estimate of orthonormal n1 x K x n3 tensor (of type Tensor)
    #   w: estimated principal components (weights) of size K x 1 x n3 (of type Tensor)
    #
    n1,K,n3 = U.shape()

    v_Omega = Tensor(Yvec.array()[idx, :, :])
    U_Omega = Tensor(U.array()[idx, :, :])

    w = tpinv(U_Omega) * v_Omega

    p = U * w
    r = v_Omega - U_Omega * w

    wnormal_F,wnorms_F = normalizeTensorVec(w)
    pnormal_F,pnorms_F = normalizeTensorVec(p)
    rnormal_F,rnorms_F = normalizeTensorVec(r)

    Ustep_F = np.zeros((n1, K, n3), dtype=complex)



    for i in range(0, int(np.ceil((n3 + 1) / 2))):
        if (step is None):
            t = np.arctan(rnorms_F[0, 0, i] / wnorms_F[0, 0, i])
            alpha = (np.cos(t) - 1)
            beta = np.sin(t)
        else:
            sG = rnorms_F[0, 0, i] * wnorms_F[0, 0, i]
            alpha = (np.cos(step * sG) - 1)
            beta = np.sin(step * sG)

        gamma = alpha * pnormal_F[:, :, i]
        gamma[idx, :] = gamma[idx, :] + beta * rnormal_F[:, :, i]
        Ustep_F[:, :, i] = gamma @ wnormal_F[:, :, i].conj().T

    for i in range(int(np.ceil((n3 + 1) / 2)), n3):
        Ustep_F[:, :, i] = np.conj(Ustep_F[:, :, n3 - i])

    Ustep = Tensor(np.real(np.fft.ifft(Ustep_F, axis=2)))

    U = U + Ustep

    orthoTest(U)

    return U,w


def normalizeTensorVec2(v_F):
    vnorms_F = np.expand_dims(np.linalg.norm(v_F, axis=0), axis=0)
    vnormal_F = v_F / vnorms_F

    return vnormal_F, vnorms_F


def fourier_dot(abar, bbar):
    n1, _, n3 = abar.shape
    _, n2, _ = bbar.shape
    cbar = np.zeros(n3, dtype=complex)

    for i in range(0, int(np.ceil((n3 + 1) / 2))): cbar[i] = abar[:, :, i].conj().T @ bbar[:, :, i]

    for i in range(int(np.ceil((n3 + 1) / 2)), n3): cbar[i] = np.conj(cbar[n3 - i])

    return np.sum(cbar)


def fourier_mult(Abar, Bbar):
    n1, _, n3 = Abar.shape
    _, n2, _ = Bbar.shape
    Cbar = np.zeros((n1, n2, n3), dtype=complex)

    for i in range(0, int(np.ceil((n3 + 1) / 2))): Cbar[:, :, i] = Abar[:, :, i] @ Bbar[:, :, i]

    for i in range(int(np.ceil((n3 + 1) / 2)), n3): Cbar[:, :, i] = np.conj(Cbar[:, :, n3 - i])

    return Cbar


def cg(Ubar, vbar, mask, cg_iter=None, tol=1e-12):

    n1,K,n3 = Ubar.shape

    if (cg_iter is None):
        cg_iter = np.prod(vbar.shape)

    UbarT = np.transpose(Ubar, [1, 0, 2]).conj()
    wbar = np.zeros((K, 1, n3), dtype=complex)
    bbar = fourier_mult(UbarT, vbar)

    F_inv = lambda qbar: np.real(np.fft.ifft(qbar, axis=2))
    F = lambda q: np.fft.fft(q, axis=2)
    A = lambda x: fourier_mult(UbarT, F(np.multiply(mask, F_inv(fourier_mult(Ubar, x)))))

    Aw = A(wbar)
    r = bbar - Aw
    p = r.copy()
    rsold = fourier_dot(r, r)

    num_iters = 0
    for i in range(0, cg_iter):

        Ap = A(p)

        # test = fourier_dot(p, Ap)
        alpha = rsold / fourier_dot(p, Ap)
        wbar = wbar + alpha * p
        r = r - alpha * Ap
        rsnew = fourier_dot(r, r)

        p = r + (rsnew / rsold) * p
        rsold = rsnew

        num_iters += 1

        if (np.sqrt(np.real(rsnew)) < tol):
            break


    return wbar, num_iters


def toucan_stream(Yvec, M, U, step=None, cgiter = None, cgtol = None):
    #
    # Description: performs one step of TOUCAN in the regime of missing random ENTRIES to estimate n1 x K x n3
    # orthonormal tensor U from input tensor column of size n1 x 1 x n3
    #
    # Inputs:
    #   Yvec: n1 x 1 x n3 tensor column (of type Tensor)
    #   Mask: n1 x 1 x n3 array of 0 and 1 where Yvec is observed on indices indicated by 1
    #   U: n1 x K x n3 orthonormal tensor (of type Tensor) initial estimate
    #
    # Outputs:
    #   U: updated estimate of orthonormal n1 x K x n3 tensor (of type Tensor)
    #   w: estimated weights of size K x 1 x n3 (of type Tensor)
    #

    M = np.expand_dims(M.astype(float), axis=1)

    n1,K,n3 = U.shape()

    v = Tensor(np.multiply(M, Yvec.array()))

    Ubar = np.fft.fft(U.array(), axis=2)
    UbarT = np.transpose(Ubar, [1, 0, 2]).conj()
    vbar = np.fft.fft(v.array(), axis=2)

    ## Update w
    if(cgtol is None):
        cgtol = 1e-14

    wbar,cg_iters = cg(Ubar, vbar, M, cg_iter = cgiter, tol=cgtol)

    ## Update U
    pbar = fourier_mult(Ubar, wbar)

    p = np.real(np.fft.ifft(pbar, axis=2))
    r = np.multiply(M, v.array() - p)
    rbar = np.fft.fft(r, axis=2)

    rbar2 = np.zeros(rbar.shape, dtype=complex)
    for i in range(0, int(np.ceil((n3 + 1) / 2))):
        rbar2[:, :, i] = Ubar[:, :, i] @ (UbarT[:, :, i] @ rbar[:, :, i])
    for i in range(int(np.ceil((n3 + 1) / 2)), n3):
        rbar2[:, :, i] = np.conj(rbar2[:, :, n3 - i])

    rbar -= rbar2

    wnormal_F, wnorms_F = normalizeTensorVec2(wbar)
    pnormal_F, pnorms_F = normalizeTensorVec2(pbar)
    rnormal_F, rnorms_F = normalizeTensorVec2(rbar)

    Ustep_F = np.zeros((n1, K, n3), dtype=complex)

    for i in range(0, int(np.ceil((n3 + 1) / 2))):
        if (step is None):
            t = np.arctan(rnorms_F[0, 0, i] / wnorms_F[0, 0, i])
            alpha = (np.cos(t) - 1)
            beta = np.sin(t)
        else:
            sG = rnorms_F[0, 0, i] * wnorms_F[0, 0, i]
            alpha = (np.cos(step * sG) - 1)
            beta = np.sin(step * sG)

        gamma = alpha * pnormal_F[:, :, i]
        gamma = gamma + beta * rnormal_F[:, :, i]
        Ustep_F[:, :, i] = gamma @ wnormal_F[:, :, i].conj().T

    for i in range(int(np.ceil((n3 + 1) / 2)), n3):
        Ustep_F[:, :, i] = np.conj(Ustep_F[:, :, n3 - i])

    Ubar = Ubar + Ustep_F

    U = Tensor(np.real(np.fft.ifft(Ubar, axis=2)))
    w = Tensor(np.real(np.fft.ifft(wbar, axis=2)))
    # orthoTest(U)

    return U, w, cg_iters


def toucan(Y,mask,rank,tube,outer,mode,tol=1e-9,step=None,cgiter=None,cgtol=1e-9,fun=lambda Lhat, idx: [0,0],
           randomOrder=False,verbose=False, U0=None):

    n1,n2,n3 = Y.shape()

    ### Initialize U
    K = rank

    if(U0 is None):
        U = tsvd(Tensor(np.random.randn(n1, K, n3)),full=False)[0]
        U = Tensor(U.array()[:, :K, :])
    else:
        U = U0

    Lhat = np.zeros((n1, n2, n3))
    What = np.zeros((K,n2,n3))

    stats_toucan = np.zeros((outer * n2 + 1,4))
    # cost,nrmse = fun(Tensor(Lhat))
    cost = 0
    nrmse = 1
    stats_toucan[0,:] = [cost,nrmse, 0, 0]

    iter = 1
    for outer in range(0, outer):
        if(nrmse < tol):
            stats_toucan = stats_toucan[:iter,:]
            break;

        if (randomOrder):
            frame_order = np.random.permutation(n2)
        else:
            frame_order = np.arange(0,n2)

        for inner in range(0, n2):
            frame_idx = frame_order[inner]
            Yvec = Tensor(np.expand_dims(Y.array()[:, frame_idx, :], axis=1))

            if (tube is True):
                idx = np.where(mask[:, inner, 0] > 0)[0]
                tStart = time.time()
                U, w = toucan_stream_tube(Yvec, idx, U, step)
                tEnd = time.time()
                cg_iters = 0
            else:
                if(cgiter is not None):
                    num_cgiter = cgiter**(outer + 1)
                else:
                    num_cgiter = None

                tStart = time.time()
                U, w, cg_iters = toucan_stream(Yvec, mask[:, frame_idx, :], U, step,cgiter = num_cgiter, cgtol=cgtol)
                tEnd = time.time()

            tElapsed = tEnd - tStart

            if(mode == 'online'):                           ## online mode
                rec = U * w
                # cost, nrmse = fun(rec.array().squeeze(),frame_idx)
                Lhat[:, frame_idx, :] = rec.array().squeeze()
        
            else:
                What[:,frame_idx,:] = w.array().squeeze()
                Lhat = (U * Tensor(What)).array()           ## batch mode

            cost, nrmse = fun(Tensor(Lhat),frame_idx)
            if(nrmse < tol):
                break;

            stats_toucan[iter, :] = [cost, nrmse, cg_iters, tElapsed]
            iter += 1

            if(verbose):
                if(inner % 10 ==0):
                    print('Outer[{:d}], Inner[{:d}]: NRMSE: {:.8f} '.format(outer, inner, nrmse))

    tElapsed = np.sum(stats_toucan[:,-1])


    return Tensor(Lhat), U, stats_toucan, tElapsed
