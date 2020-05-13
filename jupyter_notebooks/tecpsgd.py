import numpy as np
import time

def compute_cost_tensor(X, P, PA, tensor_dims):
    n1 = tensor_dims[0]
    n2 = tensor_dims[1]
    n3 = tensor_dims[2]

    Diff = P * X - PA
    Diff_flat = np.reshape(Diff, (n1 * n2, n3))

    return 0.5 * np.linalg.norm(Diff_flat, 'fro') ** 2

def TeCPSGD(A_in, Omega_in, Gamma_in, tensor_dims, rank, xinit, options, fun = lambda X: 0):
    A = A_in # Full entries
    Omega = Omega_in # Trainingset 'Omega'
    Gamma = Gamma_in # Test set 'Gamma'

    Yhat = np.zeros(A.shape)

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
    # mu              = options['mu']
    stepsize_init   = options['stepsize']
    maxepochs       = options['maxepochs']
    tolcost         = options['tolcost']
    store_subinfo   = options['store_subinfo']
    store_matrix    = options['store_matrix']
    verbose         = options['verbose']

    if options['permute_on'] is None:
        permute_on = 1
    else:
        permute_on = options['permute_on']

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

    # set parameters
    eta = 0

    if verbose > 0:
        print('TeCPSGD [{:3f}] Epoch 000, Cost {:.5f}, Cost(test) {:.5f}, Stepsize {:.5f}'.format(stepsize_init,
                                                                                              train_cost,test_cost, eta))
    # main loop
    for outiter in range(0,maxepochs):
        #permute samples
        if permute_on:
            col_order = np.random.permutation(slice_length)
        else:
            col_order = np.arange(0,slice_length)


        # Begin the time counter for the epoch

        for k in range(0,slice_length):

            tStart = time.time()

            fnum = (outiter + 1 - 1) * slice_length + k + 1

            #sampled original image
            I_mat = A[:,:, col_order[k]]
            Omega_mat = Omega[:,:, col_order[k]]
            I_mat_Omega = Omega_mat * I_mat

            # Recalculate gamma(C)
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

            L_rec = A_t0 @ np.diag(gamma) @ B_t0.T
            diff = Omega_mat * (I_mat - L_rec)

            eta = stepsize_init / (1 +lam * stepsize_init * fnum)
            A_t1 = (1 - lam * eta) * A_t0 + eta * diff @ B_t0 @ np.diag(gamma) # equation(20) & (21)
            B_t1 = (1 - lam * eta) * B_t0 + eta * diff.T @ A_t0 @ np.diag(gamma) # equation (20) & (22)

            A_t0 = A_t1
            B_t0 = B_t1

            # Recalculate gamma(C)
            temp3 = 0
            temp4 = 0
            for m in range(0, rows):
                alpha_remat = np.tile(A_t0[m, :].T, (cols, 1)).T
                alpha_beta = alpha_remat * B_t0.T
                I_row = I_mat_Omega[m, :]
                temp3 = temp3 + alpha_beta @ I_row.T

                Omega_mat_ind = np.where(Omega_mat[m, :] > 0)[0]
                alpha_beta_Omega = alpha_beta[:, Omega_mat_ind]
                temp4 = temp4 + alpha_beta_Omega @ alpha_beta_Omega.T

            temp4 = lam * np.eye(rank) + temp4
            # gamma = np.linalg.lstsq(temp4, temp3, rcond=-1)[0]  # gamma = temp4 \ temp3;
            gamma = np.linalg.inv(temp4)@temp3

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
                    print('TeCPSGD: fnum = {:3d}, cost = {:.3f}, error = {:.8f}'.format(fnum, train_cost, error))

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
            print('TeCPSGD [{:3f}] Epoch {:3d}, Cost {:.7f}, Cost(test) {:.7f}, Stepsize {:.7f}'.format(
                stepsize_init,  outiter, train_cost, test_cost, eta))

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