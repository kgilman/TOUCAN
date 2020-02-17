import numpy as np
import time

def compute_cost_tensor(X, P, PA, tensor_dims):
    n1 = tensor_dims[0]
    n2 = tensor_dims[1]
    n3 = tensor_dims[2]

    Diff = P * X - PA
    Diff_flat = np.reshape(Diff, (n1 * n2, n3))

    return 0.5 * np.linalg.norm(Diff_flat, 'fro') ** 2

def OLSTEC(A_in, Omega_in, Gamma_in, tensor_dims, rank, xinit, options,fun = lambda X: 0):
    A = A_in # Full entries
    Omega = Omega_in # Trainingset 'Omega'
    Gamma = Gamma_in # Test set 'Gamma'

    A_Omega = Omega_in * A_in #Training entries i.e., Omega_in. * A_in
    if Gamma_in is not None:
        A_Gamma = Gamma_in * A_in # Test entries i.e., Gamma_in. * A_in
    else:
        A_Gamma = []

    if xinit is None:
        A_t0 = np.random.randn(tensor_dims[0], rank)
        B_t0 = np.random.randn(tensor_dims[1], rank)
        C_t0 = np.random.randn(tensor_dims[2], rank)
    else:
        A_t0 = xinit['A']
        B_t0 = xinit['B']
        C_t0 = xinit['C']

    # set tensor size
    rows = tensor_dims[0]
    cols = tensor_dims[1]
    slice_length = tensor_dims[2]

    # set options
    lam             = options['lam']
    mu              = options['mu']
    maxepochs       = options['maxepochs']
    tolcost         = options['tolcost']
    store_subinfo   = options['store_subinfo']
    store_matrix    = options['store_matrix']
    verbose         = options['verbose']

    if options['permute_on'] is None:
        permute_on = 1
    else:
        permute_on = options['permute_on']

    if options['tw_flag'] is None:
        TW_Flag = False
    else:
        TW_Flag = options['tw_flag']

    if options['tw_len'] is None:
        TW_LEN = 10
    else:
        TW_LEN = options['tw_len']

    # prepare Rinv history buffers
    RAinv = np.tile(100 * np.eye(rank), (rows, 1))
    RBinv = np.tile(100 * np.eye(rank), (cols, 1))

    # prepare
    N_AlphaAlphaT = np.zeros((rank * rows, rank * (TW_LEN + 1)))
    N_BetaBetaT = np.zeros((rank * cols, rank * (TW_LEN + 1)))

    # prepare
    N_AlphaResi = np.zeros((rank * rows, TW_LEN + 1))
    N_BetaResi = np.zeros((rank * cols, TW_LEN + 1))

    # calculate initial cost
    Rec = np.zeros((rows, cols, slice_length))
    for k in range(0,slice_length):
        gamma = C_t0[k,:].T
        Rec[:,:,k] = A_t0 @ np.diag(gamma) @ B_t0.T

    train_cost = compute_cost_tensor(Rec, Omega, A_Omega, tensor_dims)

    if Gamma is None and A_Gamma is None:
        test_cost = compute_cost_tensor(Rec, Gamma, A_Gamma, tensor_dims)
    else:
        test_cost = 0

    # initialize infos
    infos = {
        'iter': 0,
        'train_cost': train_cost,
        'test_cost': test_cost,
        'time': [0]
    }

    # initialize sub_infos
    sub_infos = {
        'inner_iter': 0,
        'err_residual':0,
        'error': 0,
        'err_run_ave': [0],
        'global_train_cost':0,
        'global_test_cost':0,
        'times':[0]
    }

    if store_matrix:
        sub_infos['I'] = np.zeros((rows * cols, slice_length))
        sub_infos['L'] = np.zeros((rows * cols, slice_length))
        sub_infos['E'] = np.zeros((rows * cols, slice_length))

    if verbose > 0:
        print('TeCPSGD Epoch 000, Cost {:.5f}, Cost(test) {:.5f}'.format(train_cost,test_cost))
    # main loop

    A_t1 = A_t0.copy()
    B_t1 = B_t0.copy()

    Yhat = np.zeros(A_in.shape)

    for outiter in range(0,maxepochs):
        #permute samples
        if permute_on:
            col_order = np.random.permutation(slice_length)
        else:
            col_order = np.arange(0,slice_length)


        # Begin the time counter for the epoch

        for k in range(0,slice_length):

            tStart = time.time()

            #sampled original image
            I_mat = A[:,:, col_order[k]]
            Omega_mat = Omega[:,:, col_order[k]]
            I_mat_Omega = Omega_mat * I_mat

            # Calculate gamma
            temp3 = 0
            temp4 = 0
            for m in range(0,rows):
                alpha_remat = np.tile(A_t0[m,:].T, (cols,1)).T
                alpha_beta = alpha_remat * B_t0.T
                I_row = I_mat_Omega[m,:]
                temp3 = temp3 + alpha_beta @ I_row.T

                Omega_mat_ind = np.where(Omega_mat[m,:]>0)[0]
                alpha_beta_Omega = alpha_beta[:, Omega_mat_ind]
                temp4 = temp4 + alpha_beta_Omega @ alpha_beta_Omega.T

            temp4 = lam * np.eye(rank) + temp4
            gamma = np.linalg.lstsq(temp4,temp3,rcond=-1)[0]    # gamma = temp4 \ temp3;

            ## update A
            for m in range(0,rows):
                Omega_mat_ind = np.where(Omega_mat[m,:]>0)[0]
                I_row = I_mat_Omega[m,:]
                I_row_Omega = I_row[Omega_mat_ind]
                C_t0_Omega = B_t0[Omega_mat_ind,:]
                N_alpha_Omega = np.diag(gamma) @ C_t0_Omega.T
                N_alpha_alpha_t_Omega = N_alpha_Omega @ N_alpha_Omega.T

                # Calc TAinv(i.e.RAinv)
                TAinv = lam**(-1) * RAinv[m * rank: (m+1) * rank,:]
                if TW_Flag:
                    Oldest_alpha_alpha_t = N_AlphaAlphaT[m * rank :(m+1) * rank, 0:rank]
                    TAinv = np.linalg.inv(np.linalg.inv(TAinv) + N_alpha_alpha_t_Omega + (mu - lam * mu) * np.eye(rank) - lam ** TW_LEN * Oldest_alpha_alpha_t)
                else:
                    TAinv = np.linalg.inv(np.linalg.inv(TAinv) + N_alpha_alpha_t_Omega + (mu - lam * mu) * np.eye(rank))

                # Calc delta A_t0(m,:)
                recX_col_Omega = N_alpha_Omega.T @ A_t0[m,:].T
                resi_col_Omega = I_row_Omega.T - recX_col_Omega
                N_alpha_Resi_Omega = N_alpha_Omega @ np.diag(resi_col_Omega)

                N_resi_Rt_alpha = TAinv @ N_alpha_Resi_Omega
                delta_A_t0_m = np.sum(N_resi_Rt_alpha, 1)

                # Update A
                if TW_Flag:
                    # update A
                    Oldest_alpha_resi = N_AlphaResi[m * rank: (m+1) * rank, 1].T
                    A_t1[m,:] = A_t0[m,:]  - (mu - lam * mu) * A_t0[m,:] @ TAinv.T + delta_A_t0_m.T - lam ** TW_LEN @ Oldest_alpha_resi

                    # Store data
                    N_AlphaAlphaT[m * rank: (m+1) * rank, TW_LEN * rank + 1: (TW_LEN + 1) * rank] = \
                        N_alpha_alpha_t_Omega
                    N_AlphaResi[m * rank : (m+1) * rank, TW_LEN + 1] = np.sum(N_alpha_Resi_Omega, 2)
                else:
                    A_t1[m,:] = A_t0[m,:] - (mu - lam * mu) * A_t0[m,:] @ TAinv.T  + delta_A_t0_m.T

                # Store RAinv
                RAinv[m * rank : (m+1) * rank,:] = TAinv

            # Final update of A
            A_t0 = A_t1.copy()

            ## update B
            for n in range(0,cols):
                Omega_mat_ind = np.where(Omega_mat[:,n] > 0)[0]
                I_col = I_mat_Omega[:, n]
                I_col_Omega = I_col[Omega_mat_ind]
                A_t0_Omega = A_t0[Omega_mat_ind,:]
                N_beta_Omega = A_t0_Omega @ np.diag(gamma)
                N_beta_beta_t_Omega = N_beta_Omega.T @ N_beta_Omega

                # Calc TBinv(i.e.RBinv)
                TBinv = lam**(-1) * RBinv[n*rank: (n+1) * rank,:]
                if TW_Flag:
                    Oldest_beta_beta_t = N_BetaBetaT[n*rank:(n+1) * rank, 1: rank]
                    TBinv = np.linalg.inv(np.linalg.inv(TBinv) + N_beta_beta_t_Omega + (mu - lam * mu) * np.eye(rank)
                                          - lam**TW_LEN * Oldest_beta_beta_t)
                else:
                    TBinv = np.linalg.inv(np.linalg.inv(TBinv) + N_beta_beta_t_Omega + (mu - lam * mu) * np.eye(rank))

                # Calc delta B_t0(n,:)
                recX_col_Omega = B_t0[n,:] @ N_beta_Omega.T
                resi_col_Omega = I_col_Omega.T - recX_col_Omega
                N_beta_Resi_Omega = N_beta_Omega.T @ np.diag(resi_col_Omega)
                N_resi_Rt_beta = TBinv @ N_beta_Resi_Omega
                delta_C_t0_n = np.sum(N_resi_Rt_beta, 1)

                if TW_Flag:
                    # Upddate B
                    Oldest_beta_resi = N_BetaResi[n*rank:(n+1) * rank, 1].T
                    B_t1[n,:] = B_t0[n,:] - (mu - lam * mu) * B_t0[n,:] @ TBinv.T + delta_C_t0_n.T -lam ** TW_LEN \
                                                                                     * Oldest_beta_resi

                    # Store data
                    N_BetaBetaT[n*rank: (n+1) * rank, TW_LEN * rank + 1: (TW_LEN + 1) * rank] = N_beta_beta_t_Omega
                    N_BetaResi[n*rank: (n+1) * rank, TW_LEN + 1] = np.sum(N_beta_Resi_Omega, 2)
                else:
                    B_t1[n,:] = B_t0[n,:] - (mu - lam * mu) * B_t0[n,:] @ TBinv.T + delta_C_t0_n.T

                # Store RBinv
                RBinv[n*rank: (n+1) * rank,:] = TBinv

            # Final update of B
            B_t0 = B_t1.copy()

            # # Calculate gamma
            # temp3 = 0
            # temp4 = 0
            # for m in range(0, rows):
            #     alpha_remat = np.tile(A_t0[m, :].T, (cols, 1)).T
            #     alpha_beta = alpha_remat * B_t0.T
            #     I_row = I_mat_Omega[m, :]
            #     temp3 = temp3 + alpha_beta @ I_row.T

            #     Omega_mat_ind = np.where(Omega_mat[m, :] > 0)[0]
            #     alpha_beta_Omega = alpha_beta[:, Omega_mat_ind]
            #     temp4 = temp4 + alpha_beta_Omega @ alpha_beta_Omega.T

            # temp4 = lam * np.eye(rank) + temp4
            # gamma = np.linalg.lstsq(temp4, temp3, rcond=-1)[0]  # gamma = temp4 \ temp3;

            tElapsed = time.time() - tStart

            #Store gamma into C_t0
            C_t0[col_order[k],:] = gamma.T

            #Reconstruct Low - rank Matrix
            L_rec = A_t0 @ np.diag(gamma) @ B_t0.T
            Yhat[:,:,k] = L_rec

            ## Diagnostics

            if store_matrix:
                E_rec = I_mat - L_rec
                sub_infos['E'] = np.append(sub_infos['E'], np.vectorize(E_rec))
                I = sub_infos['I']
                I[:,k] = np.vectorize(I_mat_Omega)
                sub_infos['I'] = I

                L = sub_infos['L']
                L[:, k] = np.vectorize(L_rec)
                sub_infos['L'] = L

                E = sub_infos['E']
                E[:, k] = np.vectorize(E_rec)
                sub_infos['E'] = E

                # sub_infos.I[:, k] = np.vectorize(I_mat_Omega)
                # sub_infos.L[:, k] = np.vectorize(L_rec)
                # sub_infos.E[:, k] = np.vectorize(E_rec)

            if store_subinfo:
                # Residual Error
                norm_residual = np.linalg.norm(I_mat - L_rec,'fro')
                norm_I = np.linalg.norm(I_mat,'fro')
                error = norm_residual / norm_I
                sub_infos['inner_iter'] = np.append(sub_infos['inner_iter'], (outiter + 1 - 1) * slice_length + k + 1)
                sub_infos['err_residual'] = np.append(sub_infos['err_residual'], error)
                sub_infos['times'] = np.append(sub_infos['times'],tElapsed)
                sub_infos['error'] = np.append(sub_infos['error'],fun(Yhat))

                #Running - average Estimation Error
                if k == 0:
                    run_error = error
                else:
                    run_error = (sub_infos['err_run_ave'][-1] * (k + 1 - 1) + error) / (k+1)

                sub_infos['err_run_ave'] = np.append(sub_infos['err_run_ave'], run_error)

                # Store reconstruction Error
                if store_matrix:
                    E_rec = I_mat - L_rec
                    sub_infos['E'] = np.append(sub_infos['E'],np.vectorize(E_rec))

                # for f in range(0,slice_length):
                #     gamma = C_t0[f,:].T
                #     Rec[:,:, f] = A_t0 @ np.diag(gamma) @ B_t0.T

                # Global train_cost computation
                # train_cost = compute_cost_tensor(Rec, Omega, A_Omega, tensor_dims)
                # if Gamma is None and A_Gamma is None:
                #     test_cost = compute_cost_tensor(Rec, Gamma, A_Gamma, tensor_dims)
                # else:
                #     test_cost = 0

                # sub_infos['global_train_cost'] = np.append(sub_infos['global_train_cost'],train_cost)
                # sub_infos['global_test_cost'] = np.append(sub_infos['global_test_cost'],test_cost)

                if verbose:
                    train_cost = 0
                    fnum = (outiter + 1 - 1) * slice_length + k + 1
                    print('OLSTEC: fnum = {:3d}, cost = {:.3f}, error = {:.8f}'.format(fnum, train_cost, error))

        # store infos
        infos['iter'] = np.append(infos['iter'],outiter)
        # infos['time'] = np.append(infos['time'], infos['time'][-1] + time.time() - t_begin)

        if store_subinfo is False:
            # for f in range(0,slice_length):
            #     gamma = C_t0[f,:].T
            #     Rec[:,:, f] = A_t0 @ np.diag(gamma) @ B_t0.T
            pass

        # train_cost = compute_cost_tensor(Rec, Omega, A_Omega, tensor_dims)
        if Gamma is None and A_Gamma is None:
            test_cost = compute_cost_tensor(Rec, Gamma, A_Gamma, tensor_dims)
        else:
            test_cost = 0

        infos['train_cost'] = [infos['train_cost'], train_cost]
        infos['test_cost'] = [infos['test_cost'], test_cost]

        if verbose > 0:
            print('OLSTEC Epoch {:3d}, Cost {:.7f}, Cost(test) {:.7f}'.format(outiter, train_cost, test_cost))

        #stopping criteria: cost tolerance reached
        # if train_cost < tolcost:
        #     print('train_cost sufficiently decreased.')
        #     break

    Xsol = {
        'A': A_t0,
        'B': B_t0,
        'C': C_t0
    }

    return Xsol, Yhat, infos, sub_infos