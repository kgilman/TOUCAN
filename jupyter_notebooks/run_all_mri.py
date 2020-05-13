import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tsvd import *
import scipy.io as sio
from stc import *
from olstec import *
from tecpsgd import *
from skimage.measure import compare_ssim



def im2kspace(images):
    ### images are nt x nx x ny
    c1 = np.sqrt(n2)
    c2 = np.sqrt(n3)

    # c1 = 1
    # c2 = 1
    z = np.fft.fftshift(np.fft.fft(np.fft.fftshift(images, axes=1), axis=1), axes=1) * c1
    kdata = np.fft.fftshift(np.fft.fft(np.fft.fftshift(z, axes=2), axis=2), axes=2) * c2
    return kdata


def kspace2im(kdata):
    ### kdata is nt x nx x ny

    # data_rec = tensor.array() + 1j * C.array()
    # data_rec = np.transpose(data_rec, (1, 0, 2))
    n1, n2, n3 = kdata.shape
    c = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(kdata, axes=1), axis=1), axes=1) * 1 / np.sqrt(n2)
    im_rec = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(c, axes=2), axis=2), axes=2) * 1 / np.sqrt(n3)
    # c = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(kdata, axes=1), axis=1), axes=1)
    # im_rec = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(c, axes=2), axis=2), axes=2)
    return np.abs(im_rec)


def compute_stats(Rhat, Chat, R, C, orig_im, kdata_orig):
    slice_err = []
    for i in range(0, n2):
        slice_err.append(tfrobnorm_array(Rhat.array()[:, i, :] - R.array()[:, i, :]) / tfrobnorm_array(R.array()[:, i,
                                                                                                       :]))

    kdata_rec = Rhat.array() + 1j * Chat.array()
    kdata_rec = np.transpose(kdata_rec, (1, 0, 2))
    im_rec = kspace2im(kdata_rec)
    rec_im = np.abs(im_rec)

    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(orig_im[vis_idx, :, :], cmap='gray')
    plt.title('Original magnitude image')

    plt.subplot(2, 2, 2)
    test = np.log(np.abs(kdata_orig[vis_idx, :, :]) + 1e-13)
    plt.imshow(test, cmap='gray')
    plt.title('Original Magnitude kspace')

    plt.subplot(2, 2, 3)
    plt.imshow(rec_im[vis_idx, :, :], cmap='gray')
    plt.title('Reconstructed')
    plt.subplot(2, 2, 4)
    plt.imshow(np.log(np.abs((kdata_rec[vis_idx, :, :])) + 1e-13), cmap='gray')
    plt.title('Reconstructed Magnitude kspace')
    plt.show()
    plt.close()

    return slice_err, rec_im, kdata_rec
# plt.ioff()
datasets = ['brain','invivo_cardiac']
# datasets = ['brain']
# datasets = ['invivo_cardiac']
# dataset = 'brain'
# dataset = 'invivo_cardiac'
# datasets = ['aperiodic_pincat']
vis_idx = 30
for dataset in datasets:

    if(dataset is 'brain'):
        real_filepath = '/Users/kgilman/Desktop/t-SVD/real_mri.mat'
        imag_filepath = '/Users/kgilman/Desktop/t-SVD/imag_mri.mat'

        real_data = sio.loadmat(real_filepath)['real_mri']
        imag_data = sio.loadmat(imag_filepath)['imag_mri']

        kdata = real_data + 1j*(imag_data)
        kdata = np.transpose(kdata,(2,0,1))
        orig_im = kspace2im(kdata)
        fig = plt.figure()
        plt.imshow(orig_im[10,:,:],cmap='gray')
        plt.show()
        plt.close(fig)
    else:
        if(dataset is 'invivo_cardiac'):
            data = sio.loadmat('/Users/kgilman/Desktop/t-SVD/invivo_perfusion.mat')['x']
        else:
            data = sio.loadmat('/Users/kgilman/Desktop/t-SVD/aperiodic_pincat.mat')['new']
        data = np.transpose(data,(2,0,1))
        n1,n2,n3 = data.shape
        # orig_im = np.abs(data)
        orig_im = (data)
        ## convert to kspace
        kdata = im2kspace(data)

    n1,n2,n3 = kdata.shape
    ### Form the real and imag tensors
    kdata_orig = kdata.copy()
    kdata = np.transpose(kdata, (1, 0, 2))

    R = np.real(kdata)
    Rmean = np.mean(R)

    C = np.imag(kdata)
    Cmean = np.mean(C)

    # R -= Rmean
    # C -= Cmean

    R = Tensor(R)
    C = Tensor(C)

    n1, n2, n3 = R.shape()


    def computeStats(im):
        plt.subplot(1,2,1)
        plt.imshow(im[30,:,:],cmap='gray')
        plt.subplot(1,2,2)
        plt.imshow(orig_im[30,:,:],cmap='gray')
        plt.show()
        nrmse = tfrobnorm_array(im - orig_im) / tfrobnorm_array(orig_im)
        scores = []
        for i in range(im.shape[0]):
            (score, diff) = compare_ssim(orig_im[i, :, :] / np.max(orig_im[i, :, :]),
                                         im[i, :, :] / np.max(im[i, :, :]), full=True)
            scores.append(score)
        return nrmse, np.mean(scores)

    ### Generate k-space sampling mask
    # tubes = [True, False]
    tubes = [True, False]
    rhos = [0.8, 0.6, 0.5]
    # rhos = [0.8]
    # rhos = [0.4]
    for tube in tubes:
        for rho in rhos:
            np.random.seed(0)
            if (tube is False):
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

            Rfrob = tfrobnorm(R)
            Cfrob = tfrobnorm(C)
            R_sample = Tensor(R.array() * mask)
            C_sample = Tensor(C.array() * mask)

            kdata_rec = R_sample.array() + 1j * C_sample.array()
            kdata_rec = np.transpose(kdata_rec, (1, 0, 2))
            im_rec = kspace2im(kdata_rec)
            # subsampled_im = np.abs(im_rec[vis_idx,:,:])
            subsampled_im = np.abs(im_rec)

            # plt.figure(figsize=(15,12))
            fig = plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(subsampled_im[vis_idx, :, :], cmap='gray')
            plt.title('Reconstruction from subsampled')
            plt.subplot(1, 2, 2)
            plt.imshow(np.log(np.abs((kdata_rec[vis_idx, :, :])) + 1e-8), cmap='gray')
            plt.title('Magnitude of subsampled kspace')
            plt.rcParams.update({'font.size': 22})
            name = '/Users/kgilman/Desktop/t-SVD/mri_reconstruct/mri_results/subsampled_and_kspace_tube_' + str(tube) + '_' + str(int(rho * 100)) + '.eps'
            plt.savefig(name)
            plt.show()
            plt.close(fig)

            ############################# TNN-ADMM #################################

            niter = 300
            # niter = 100

            fun = lambda X: [0, tfrobnorm(X - R) / Rfrob]
            Rhat_tnn, stats, tElapsed_tensor = lrtc(R_sample, mask, rho=1.1, niter=niter, min_iter = 100, fun=fun, verbose=True)

            fun = lambda X: [0, tfrobnorm(X - C) / Cfrob]
            Chat_tnn, stats, tElapsed_tensor = lrtc(C_sample, mask, rho=1.1, niter=niter, min_iter = 100, fun=fun, verbose=True)

            cost_tensor = stats[:, 0]
            nrmse_tensor = stats[:, 1]
            times_tensor = stats[:, 2]

            print('Time elapsed: Tensor: {:.3f} '.format(tElapsed_tensor))
            print('Final R NRMSE: Tensor: {:.8f} '.format(tfrobnorm(Rhat_tnn - R) / Rfrob))
            print('Final C NRMSE: Tensor: {:.8f} '.format(tfrobnorm(Chat_tnn - C) / Cfrob))

            slice_err_tnn, rec_im_tnn, kdata_rec_tnn = compute_stats(Rhat_tnn, Chat_tnn, R, C, orig_im, kdata_orig)

            #############################################################3
            ## TCTF
            # niter = 200
            niter = 300

            if(dataset is 'brain'):
                rank = 1
            else:
                rank = 5

            fun = lambda U, V: [0, tfrobnorm(U * V - R) / Rfrob]
            Xtctf, Ztctf, stats_tctf, tElapsed_tctf = tctf(R_sample, mask, rank=rank, niter=niter, min_iter = 50, fun=fun,
                                                           verbose=False)
            Rhat_tctf = Xtctf * Ztctf

            fun = lambda U, V: [0, tfrobnorm(U * V - C) / Cfrob]
            Xtctf, Ztctf, stats_tctf, tElapsed_tctf = tctf(C_sample, mask, rank=rank, niter=niter, min_iter = 50, fun=fun,
                                                           verbose=False)
            Chat_tctf = Xtctf * Ztctf

            nrmse_tctf = stats_tctf[:, 1]
            times_tctf = stats_tctf[:, -1]
            print('TCTF Time: {:4f}'.format(tElapsed_tctf))
            print('Final R NRMSE: Tensor: {:.8f} '.format(tfrobnorm(Rhat_tctf - R) / Rfrob))
            print('Final C NRMSE: Tensor: {:.8f} '.format(tfrobnorm(Chat_tctf - C) / Cfrob))

            slice_err_tctf, rec_im_tctf, kdata_rec_tctf = compute_stats(Rhat_tctf, Chat_tctf, R, C, orig_im, kdata_orig)
            #
            # #############################################################3
            ## TOUCAN
            if (dataset is 'brain'):
                rank = 1
            else:
                rank = 5
            outer = 1
            # fun = lambda X: [0, tfrobnorm(X - R) / Rfrob]

            fun = lambda X, k: [0, tfrobnorm_array(X.array()[:, k, :] - R.array()[:, k, :]) / tfrobnorm_array(
                R.array()[:, k, :])]

            Rhat_toucan, U, stats_toucan, tElapsed_toucan = toucan(R_sample, mask, rank, tube=tube, mode='online',
                                                                   outer=outer,
                                                                   fun=fun, cgtol=1e-7, randomOrder=False,
                                                                   verbose=False)

            fun = lambda X, k: [0, tfrobnorm_array(X.array()[:, k, :] - C.array()[:, k, :]) / tfrobnorm_array(
                C.array()[:, k, :])]
            Chat_toucan, U, stats_toucan, tElapsed_toucan = toucan(C_sample, mask, rank, tube=tube, mode='online',
                                                                   outer=outer,
                                                                   fun=fun, cgtol=1e-7, randomOrder=False,
                                                                   verbose=False)

            print('Initial R NRMSE: Tensor: {:.8f} '.format(tfrobnorm(R_sample - R) / Rfrob))
            print('Initial C NRMSE: Tensor: {:.8f} '.format(tfrobnorm(C_sample - C) / Cfrob))
            print('Final R NRMSE: Tensor: {:.8f} '.format(tfrobnorm(Rhat_toucan - R) / Rfrob))
            print('Final C NRMSE: Tensor: {:.8f} '.format(tfrobnorm(Chat_toucan - C) / Cfrob))

            slice_err_toucan, rec_im_toucan, kdata_rec_toucan = compute_stats(Rhat_toucan, Chat_toucan, R, C, orig_im,
                                                                              kdata_orig)

            #
            # #############################################################3
            #### STC

            if(tube is False):

                Tensor_R_sample = np.transpose(R_sample.array(), [0, 2, 1])
                Tensor_C_sample = np.transpose(C_sample.array(), [0, 2, 1])
                Mask_Y = np.transpose(mask, [0, 2, 1])
                numcycles = 1
                outer = 1
                r1 = 25
                r2 = 25
                # r3 = 1
                if (dataset is 'brain'):
                    r3 = 1
                else:
                    r3 = 5
                fun = lambda Lhat, idx: [0, 1]
                Rhat_stc, stats, tElapsed_stc = stc(Tensor_R_sample, Mask_Y, r1, r2, r3, outer, numcycles, fun=fun,
                                                    verbose=False)
                Chat_stc, stats, tElapsed_stc = stc(Tensor_C_sample, Mask_Y, r1, r2, r3, outer, numcycles, fun=fun,
                                                    verbose=False)

                Rhat_stc = Tensor(np.transpose(Rhat_stc, [0, 2, 1]))
                Chat_stc = Tensor(np.transpose(Chat_stc, [0, 2, 1]))

                Rhat_nrmse_stc = tfrobnorm((Rhat_stc) - R) / Rfrob
                Chat_nrmse_stc = tfrobnorm((Chat_stc) - C) / Cfrob

                print('STC Time: {:4f}'.format(tElapsed_stc))
                print('Rhat NRMSE STC: {:6f}'.format(Rhat_nrmse_stc))
                print('Chat NRMSE STC: {:6f}'.format(Chat_nrmse_stc))

                slice_err_stc, rec_im_stc, kdata_rec_stc = compute_stats(Rhat_stc, Chat_stc, R, C, orig_im, kdata_orig)

            ######################################################
            ### OLSTEC

            rank = 50

            Tensor_R_sample = np.transpose(R.array(), [0, 2, 1])
            Tensor_C_sample = np.transpose(C.array(), [0, 2, 1])
            Mask_Y = np.transpose(mask, [0, 2, 1])

            tensor_dims = [n1, n3, n2]
            maxepochs = 1
            tolcost = 1e-14
            permute_on = False


            if (dataset is 'brain'):
                options = {
                'maxepochs': maxepochs,
                'tolcost': tolcost,
                'lam': 0.8,
                'mu': 0.001,
                'permute_on':  permute_on,
                'store_subinfo': True,
                'store_matrix': False,
                'verbose': False,
                'tw_flag': None,
                'tw_len': None
                }
            else:
                options = {
                    'maxepochs': maxepochs,
                    'tolcost': tolcost,
                    'lam': 0.5,
                    'mu': 0.0001,
                    'permute_on': permute_on,
                    'store_subinfo': True,
                    'store_matrix': False,
                    'verbose': False,
                    'tw_flag': None,
                    'tw_len': None
                }

            Xinit = {
                'A': np.random.randn(tensor_dims[0], rank),
                'B': np.random.randn(tensor_dims[1], rank),
                'C': np.random.randn(tensor_dims[2], rank)
            }

            Xsol_olstec, Rhat_olstec, info_olstec, sub_infos_olstec = OLSTEC(Tensor_R_sample, Mask_Y, None, tensor_dims,
                                                                             rank,
                                                                             Xinit, options)
            Xsol_olstec, Chat_olstec, info_olstec, sub_infos_olstec = OLSTEC(Tensor_C_sample, Mask_Y, None, tensor_dims,
                                                                             rank,
                                                                             Xinit, options)

            Rhat_olstec = Tensor(np.transpose(Rhat_olstec, [0, 2, 1]))
            Chat_olstec = Tensor(np.transpose(Chat_olstec, [0, 2, 1]))

            Rhat_nrmse_olstec = tfrobnorm(Rhat_olstec - R) / Rfrob
            Chat_nrmse_olstec = tfrobnorm(Chat_olstec - C) / Cfrob
            tElapsed_olstec = np.sum(sub_infos_olstec['times'])

            print('OLSTEC Time: {:4f}'.format(tElapsed_olstec))
            print('Rhat NRMSE OLSTEC: {:6f}'.format(Rhat_nrmse_olstec))
            print('Chat NRMSE OLSTEC: {:6f}'.format(Chat_nrmse_olstec))

            slice_err_olstec, rec_im_olstec, kdata_rec_olstec = compute_stats(Rhat_olstec, Chat_olstec, R, C, orig_im,
                                                                              kdata_orig)
            #######################################################################3
            ##### TeCPSGD
            rank = 50
            tensor_dims = [n1, n3, n2]
            maxepochs = 1
            tolcost = 1e-14
            permute_on = False

            if (dataset is 'brain'):
                options = {
                'maxepochs': maxepochs,
                'tolcost': tolcost,
                'lam': 0.001,
                'stepsize': 0.01,
                'permute_on':  permute_on,
                'store_subinfo': True,
                'store_matrix': False,
                'verbose': False
                }

            else:
                options = {
                    'maxepochs': maxepochs,
                    'tolcost': tolcost,
                    'lam': 0.0001,
                    'stepsize': 100000,
                    'permute_on': permute_on,
                    'store_subinfo': True,
                    'store_matrix': False,
                    'verbose': False
                }

            # Xinit = {
            #     'A': np.random.randn(tensor_dims[0], rank),
            #     'B': np.random.randn(tensor_dims[1], rank),
            #     'C': np.random.randn(tensor_dims[2], rank)
            # }

            Xsol_TeCPSGD, Rhat_tecpsgd, info_TeCPSGD, sub_infos_TeCPSGD = TeCPSGD(Tensor_R_sample, Mask_Y, None,
                                                                                  tensor_dims, rank,
                                                                                  Xinit, options)

            Xsol_TeCPSGD, Chat_tecpsgd, info_TeCPSGD, sub_infos_TeCPSGD = TeCPSGD(Tensor_C_sample, Mask_Y, None,
                                                                                  tensor_dims, rank,
                                                                                  Xinit, options)

            Rhat_tecpsgd = Tensor(np.transpose(Rhat_tecpsgd, [0, 2, 1]))
            Chat_tecpsgd = Tensor(np.transpose(Chat_tecpsgd, [0, 2, 1]))

            Rhat_nrmse_tecpsgd = tfrobnorm(Rhat_tecpsgd - R) / Rfrob
            Chat_nrmse_tecpsgd = tfrobnorm(Chat_tecpsgd - C) / Cfrob

            print('Rhat NRMSE TeCPSGD: {:6f}'.format(Rhat_nrmse_tecpsgd))
            print('Chat NRMSE TeCPSGD: {:6f}'.format(Chat_nrmse_tecpsgd))
            tElapsed_tecpsgd = np.sum(sub_infos_TeCPSGD['times'])

            slice_err_tecpsgd, rec_im_tecpsgd, kdata_rec_tecpsgd = compute_stats(Rhat_tecpsgd, Chat_tecpsgd, R, C,
                                                                                 orig_im, kdata_orig)

            ##############################################


            #     print(name + ' NRMSE: {:.5f}'.format(nrmse))
            #     print(name + ' SSIM: {:.5f} \n'.format(score))

            vis_idx = 39
            fig = plt.figure()
            plt.figure(figsize=(20, 20))
            plt.subplot(2, 4, 1)
            plt.imshow(orig_im[vis_idx, :, :], cmap='gray')
            # plt.title('Original: NRMSE: {:5f}, SSIM {:5f}'.format(*computeStats(orig_im)))
            plt.title('Original')

            plt.subplot(2, 4, 2)
            plt.imshow(subsampled_im[vis_idx, :, :], cmap='gray')
            # plt.title('Subsampled: NRMSE: {:5f}, SSIM {:5f}'.format(*computeStats(subsampled_im)))
            plt.title('Subsampled')

            plt.subplot(2, 4, 3)
            plt.imshow(rec_im_toucan[vis_idx, :, :], cmap='gray')
            # plt.title('TOUCAN: NRMSE: {:5f}, SSIM {:5f}'.format(*computeStats(toucan_rec_im)))
            plt.title('TOUCAN')

            plt.subplot(2, 4, 4)
            plt.imshow(rec_im_tnn[vis_idx, :, :], cmap='gray')
            # plt.title('TNN-ADMM: NRMSE: {:5f}, SSIM {:5f}'.format(*computeStats(tnn_rec_im)))
            plt.title('TNN-ADMM')

            plt.subplot(2, 4, 5)
            plt.imshow(rec_im_tctf[vis_idx, :, :], cmap='gray')
            # plt.title('TCTF NRMSE: {:5f}, SSIM {:5f}'.format(*computeStats(tctf_rec_im)))
            plt.title('TCTF')

            if(tube is False):
                plt.subplot(2,4,6)
                plt.imshow(rec_im_stc[vis_idx,:,:],cmap='gray')
                # plt.title('STC NRMSE: {:5f}, SSIM {:5f}'.format(*computeStats(stc_rec_im)))
                plt.title('STC')

            plt.subplot(2, 4, 7)
            plt.imshow(rec_im_olstec[vis_idx, :, :], cmap='gray')
            # plt.title('STC NRMSE: {:5f}, SSIM {:5f}'.format(*computeStats(stc_rec_im)))
            plt.title('OLSTEC')

            plt.subplot(2, 4, 8)
            plt.imshow(rec_im_tecpsgd[vis_idx, :, :], cmap='gray')
            # plt.title('STC NRMSE: {:5f}, SSIM {:5f}'.format(*computeStats(stc_rec_im)))
            plt.title('TeCPSGD')

            plt.rcParams.update({'font.size': 22})
            name = '/Users/kgilman/Desktop/t-SVD/mri_reconstruct/mri_results/tensor_mri_reconstruct_' + dataset + '_tube' + str(tube) + '_' + str(int(rho * 100)) + \
                   '.eps'
            plt.savefig(name)
            plt.show()
            plt.close()

            fig = plt.figure(figsize=(12, 5), tight_layout=True)
            # plt.semilogy(np.arange(0, len(slice_err_tnn)), slice_err_tnn, 'r', label='TNN-ADMM')
            plt.semilogy(np.arange(0, len(slice_err_toucan)), slice_err_toucan, '#ff7f0e', label='TOUCAN')
            # plt.semilogy(np.arange(0, len(slice_err_tctf)), slice_err_tctf, 'k', label='TCTF')
            if(tube is False):
                plt.semilogy(np.arange(0, len(slice_err_stc)), slice_err_stc, '#FF007F', label='STC')
            plt.semilogy(np.arange(0, len(slice_err_olstec)), slice_err_olstec, '#8B008B', label='OLSTEC')
            plt.semilogy(np.arange(0, len(slice_err_tecpsgd)), slice_err_tecpsgd, '#00FFFF', label='TeCPSGD')
            plt.legend(bbox_to_anchor=(1.5, 1))
            plt.title('NRMSE by frame')
            plt.xlabel('Frame idx')
            name = '/Users/kgilman/Desktop/t-SVD/mri_reconstruct/mri_results/tensor_mri_reconstruct_' + dataset + '_tube_' + str(tube) + \
                   '_' + str(int(rho * 100)) + '_frameNRMSE.eps'
            plt.savefig(name)
            plt.show()
            plt.close()

            ### Print computation times
            filename = '/Users/kgilman/Desktop/t-SVD/mri_reconstruct/mri_results/times_' + dataset + '_tube_' + str(tube) + '_' + str(int(rho * 100)) + '.text'
            print('TNN ADMM: {:.3f} '.format(tElapsed_tensor), file=open(filename, "a"))
            print('TCTF: {:4f}'.format(tElapsed_tctf), file=open(filename, "a"))
            print('TOUCAN: {:4f}'.format(tElapsed_toucan), file=open(filename, "a"))
            if(tube is False):
                print('STC: {:4f}'.format(tElapsed_stc),file=open(filename, "a"))
            print('OLSTEC: {:4f}'.format(tElapsed_olstec), file=open(filename, "a"))
            print('TeCPSGD: {:4f}'.format(tElapsed_tecpsgd), file=open(filename, "a"))

            filename = '/Users/kgilman/Desktop/t-SVD/mri_reconstruct/mri_results/stats_' + dataset + '_tube_' + str(tube) + '_' + str(int(rho * 100)) + '.text'
            print('Original: NRMSE: {:5f}, SSIM {:5f}'.format(*computeStats(orig_im)), file=open(filename, "a"))
            print('Subsampled: NRMSE: {:5f}, SSIM {:5f}'.format(*computeStats(subsampled_im)), file=open(filename, "a"))
            print('TOUCAN: NRMSE: {:5f}, SSIM {:5f}'.format(*computeStats(rec_im_toucan)), file=open(filename, "a"))
            print('TNN-ADMM: NRMSE: {:5f}, SSIM {:5f}'.format(*computeStats(rec_im_tnn)), file=open(filename, "a"))
            print('TCTF NRMSE: {:5f}, SSIM {:5f}'.format(*computeStats(rec_im_tctf)), file=open(filename, "a"))
            if(tube is False):
                print('STC NRMSE: {:5f}, SSIM {:5f}'.format(*computeStats(rec_im_stc)),file=open(filename,"a"))
            print('OLSTEC NRMSE: {:5f}, SSIM {:5f}'.format(*computeStats(rec_im_olstec)), file=open(filename, "a"))
            print('TeCPSGD NRMSE: {:5f}, SSIM {:5f}'.format(*computeStats(rec_im_tecpsgd)), file=open(filename, "a"))




