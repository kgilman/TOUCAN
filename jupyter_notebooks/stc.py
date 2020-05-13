import numpy as np
import time
from numpy import ndarray

def grouse_stream(v, xIdx, U, step=None):

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


def stc_stream(Yvec, M, U1, U2, U3, NumCycles):

    V = np.multiply(M, Yvec)

    t,o = V.shape

    ## Mode 1
    for k in range(0,NumCycles):
        for i in range(0,t):
            idx = np.where(M[i,:] > 0)[0]
            U1, w1 = grouse_stream(V[i,:],idx,U1)

    W1 = np.linalg.pinv(U1) @ Yvec.T

    ## Mode 2
    for k in range(0, NumCycles):
        for j in range(0, o):
            idx = np.where(M[:, j] > 0)[0]
            U2, w2 = grouse_stream(V[:, j], idx, U2)

    W2 = np.linalg.pinv(U2) @ Yvec

    ## Mode 3
    for k in range(0,NumCycles):
        idx = np.where(np.reshape(M,-1) > 0)[0]
        U3,W3 = grouse_stream(np.reshape(Yvec,-1),idx,U3)


    return U1,U2,U3,W1,W2,W3


def stc(Y,mask,r1,r2,r3,outercycles,numcycles,tol=1e-9,fun=lambda Lhat: [0,0],randomOrder=False,verbose=False):

    t,o,d = Y.shape

    Lhat = np.zeros(Y.shape)

    ### Initialize U
    U1 = np.linalg.svd(np.random.randn(o,r1),full_matrices=False)[0]
    U2 = np.linalg.svd(np.random.randn(t,r2),full_matrices=False)[0]
    U3 = np.linalg.svd(np.random.randn(o*t,r3),full_matrices=False)[0]

    stats = np.zeros((outercycles * d + 1,3))
    # cost,nrmse = fun(Y)
    cost = 0
    nrmse = 1
    stats[0,:] = [cost,nrmse,0]

    iter = 1
    for outer in range(0, outercycles):
        if (randomOrder):
            frame_order = np.random.permutation(d)
        else:
            frame_order = np.arange(0,d)

        for inner in range(0, d):
            frame_idx = frame_order[inner]
            Yvec = Y[:, :, frame_idx]

            tStart = time.time()
            U1,U2,U3,W1,W2,W3 = stc_stream(Yvec, mask[:,:,frame_idx], U1, U2, U3, numcycles)
            tEnd = time.time()

            tElapsed = tEnd - tStart

            rec1 = U1 @ W1
            rec2 = U2 @ W2
            rec3 = U3 @ W3

            frec1 = np.transpose(rec1)
            frec2 = rec2
            frec3 = np.reshape(rec3,Yvec.shape)

            Lvec_hat = 1/3 * (frec1 + frec2 + frec3)
            Lhat[:,:,frame_idx] = Lvec_hat
            cost, nrmse = fun(Lhat,frame_idx)
            if(nrmse < tol):
                break;

            stats[iter, :] = [cost, nrmse, tElapsed]
            iter += 1

            if(verbose):
                if(inner % 10 == 0):
                    print('Outer[{:d}], Inner[{:d}]: NRMSE: {:.8f} '.format(outer, inner, nrmse))

    tElapsed = np.sum(stats[:,-1])
    return Lhat, stats, tElapsed
