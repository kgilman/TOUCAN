import numpy as np
import matplotlib.pyplot as plt
from tsvd import *
import scipy.io as sio

#
# c = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(data,axes=1),axis=1),axes=1)
# img = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(c,axes=2),axis=2),axes=2)
#
# # plt.imshow(np.abs(img[10,:,:]),cmap='gray')
# # plt.show()

def im2kspace(images):
    ### images are nt x nx x ny
    c = np.fft.fftshift(np.fft.fft(np.fft.fftshift(images, axes=1), axis=1), axes=1) * np.sqrt(n2)
    kdata = np.fft.fftshift(np.fft.fft(np.fft.fftshift(c, axes=2), axis=2), axes=2) * np.sqrt(n3)
    return kdata

def kspace2im(kdata):
    ### kdata is nt x nx x ny

    # data_rec = tensor.array() + 1j * C.array()
    # data_rec = np.transpose(data_rec, (1, 0, 2))
    n1, n2, n3 = kdata.shape
    c = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(kdata, axes=1), axis=1), axes=1) * 1 / np.sqrt(n2)
    im_rec = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(c, axes=2), axis=2), axes=2) * 1 / np.sqrt(n3)
    return im_rec

vis_idx = 69
## Read in the images
# data = sio.loadmat('/Users/kgilman/Desktop/t-SVD/invivo_perfusion.mat')['x']
# data = np.transpose(data,(2,0,1))
# n1,n2,n3 = data.shape

real_filepath = '/Users/kgilman/Desktop/t-SVD/real_mri.mat'
imag_filepath = '/Users/kgilman/Desktop/t-SVD/imag_mri.mat'

real_data = sio.loadmat(real_filepath)['real_mri']
imag_data = sio.loadmat(imag_filepath)['imag_mri']

kdata = real_data + 1j*(imag_data)

kdata = np.transpose(kdata,(2,0,1))

n1,n2,n3 = kdata.shape

test = np.log(np.abs((kdata[vis_idx,:,:])) + 1e-13)
plt.imshow(test,cmap='gray')
plt.title('Abs kspace')
plt.show()

orig_ims = kspace2im(kdata)
plt.imshow(np.abs(orig_ims[vis_idx,:,:]))
plt.title('Original magnitude image')
plt.show()

## convert to image space


# orig_im = np.abs(data[vis_idx,:,:])
# plt.imshow(orig_im)
# plt.title('Original magnitude image')
# plt.show()

# ## convert to kspace
# kdata = im2kspace(data)
#
# test = np.log(np.abs((kdata[vis_idx,:,:])) + 1e-13)
# plt.imshow(test,cmap='gray')
# plt.title('Abs kspace')
# plt.show()

### Form the real and imag tensors
kdata = np.transpose(kdata,(1,0,2))
R = Tensor(np.real(kdata))
C = Tensor(np.imag(kdata))

n1,n2,n3 = R.shape()

U,S,V = tsvd(R,full=False)
r_svals = np.diag(S.array()[:,:,0])
power80 = 0.8*np.sum(r_svals)
cum_sum = 0
k = 0
for i in range(0,len(r_svals)):
    cum_sum += r_svals[i]
    k += 1
    if(cum_sum > power80):
        break
print('Number of Real LR t-SVD Approx Components: {:.3f}'.format(k))

U,S,V = tsvd(C,full=False)
c_svals = np.diag(S.array()[:,:,0])
power80 = 0.8*np.sum(c_svals)
cum_sum = 0
k = 0
for i in range(0,len(c_svals)):
    cum_sum += c_svals[i]
    k += 1
    if(cum_sum > power80):
        break
print('Number of Complex LR t-SVD Approx Components: {:.3f}'.format(k))

plt.rcParams.update({'font.size': 22})

plt.figure(figsize=(8,5),tight_layout=True)
plt.semilogy(np.arange(0,len(r_svals)),r_svals,
        np.arange(0,len(c_svals)),c_svals)
plt.xticks(np.arange(0,len(c_svals),step=10))
plt.legend(['Real','Complex'])
plt.title('Tubal singular values')
plt.xlabel('Singular value: 80% power: {:d}'.format(k))
plt.show()

tube = True
rho = 0.2  # percentage of missing entries

print('Maximum value of R {:6f}'.format(np.max(R.array())))
print('Minimum value of R {:6f}'.format(np.min(R.array())))

np.random.seed(0)
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

Rfrob = tfrobnorm(R)
Cfrob = tfrobnorm(C)
R_sample = Tensor(R.array() * mask)
C_sample = Tensor(C.array() * mask)

kdata_rec = R_sample.array() + 1j* C_sample.array()
kdata_rec = np.transpose(kdata_rec,(1,0,2))

im_rec = kspace2im(kdata_rec)
subsampled_im = np.abs(im_rec[vis_idx,:,:])
plt.imshow(subsampled_im,cmap='gray')
plt.title('Reconstructed from subsampled')
plt.show()


############################# TNN-ADMM #################################

niter = 150

fun = lambda X: [0, tfrobnorm(X - R) / Rfrob]
Rhat_tnn, stats, tElapsed_tensor = lrtc(R_sample, mask, rho = 1.1, niter=niter, fun=fun,verbose=True)

fun = lambda X: [0, tfrobnorm(X - C) / Cfrob]
Chat_tnn, stats, tElapsed_tensor = lrtc(C_sample, mask, rho = 1.1, niter=niter, fun=fun,verbose=True)


cost_tensor = stats[:,0]
nrmse_tensor = stats[:,1]
times_tensor = stats[:,2]

print('Time elapsed: Tensor: {:.3f} '.format(tElapsed_tensor))
print('Final R NRMSE: Tensor: {:.8f} '.format(tfrobnorm(Rhat_tnn - R) / Rfrob))
print('Final C NRMSE: Tensor: {:.8f} '.format(tfrobnorm(Chat_tnn - C) / Cfrob))

kdata_rec = Rhat_tnn.array() + 1j * Chat_tnn.array()
kdata_rec = np.transpose(kdata_rec,(1,0,2))
im_rec = kspace2im(kdata_rec)
tnn_rec_im = np.abs(im_rec[vis_idx,:,:])

# plt.subplot(1,2,1)
# plt.semilogy(cost_tensor)
# plt.xlabel('Iteration')
# plt.title('Cost function')
# plt.subplot(1,2,2)
# plt.semilogy(nrmse_tensor)
# plt.xlabel('Iteration')
# plt.title('NRMSE')
# plt.show()

slice_err_tnn = []
for i in range(0,n2):
    slice_err_tnn.append(tfrobnorm_array(Rhat_tnn.array()[:,i,:] - R.array()[:,i,:]) / tfrobnorm_array(R.array()[:,i,
                                                                                                       :]))
#
#############################################################3
## TCTF

fun = lambda U,V: [0, tfrobnorm(U*V - R) / Rfrob]
Xtctf,Ztctf, stats_tctf, tElapsed_tctf = tctf(R_sample,mask,rank=3,niter = 200,fun=fun,verbose=True)
Rhat_tctf = Xtctf * Ztctf

fun = lambda U,V: [0, tfrobnorm(U*V - C) / Cfrob]
Xtctf,Ztctf, stats_tctf, tElapsed_tctf = tctf(C_sample,mask,rank=3,niter = 200,fun=fun,verbose=True)
Chat_tctf = Xtctf * Ztctf

nrmse_tctf = stats_tctf[:,1]
times_tctf = stats_tctf[:,-1]
print('TCTF Time: {:4f}'.format(tElapsed_tctf))
print('Final R NRMSE: Tensor: {:.8f} '.format(tfrobnorm(Rhat_tctf - R) / Rfrob))
print('Final C NRMSE: Tensor: {:.8f} '.format(tfrobnorm(Chat_tctf - C) / Cfrob))

# plt.semilogy(nrmse_tctf)
# plt.xlabel('Iteration')
# plt.title('NRMSE')
# plt.show()

slice_err_tctf = []
for i in range(0,n2):
    slice_err_tctf.append(tfrobnorm_array(Rhat_tctf.array()[:,i,:] - R.array()[:,i,:]) / tfrobnorm_array(R.array()[:,i,
                                                                                                       :]))

kdata_rec = Rhat_tctf.array() + 1j * Chat_tctf.array()
kdata_rec = np.transpose(kdata_rec,(1,0,2))
im_rec = kspace2im(kdata_rec)
tctf_rec_im = np.abs(im_rec[vis_idx,:,:])

####################################################################

# test = np.log(np.abs(R_sample.array()[:,vis_idx,:]) + 1e-10)
# plt.imshow(test,cmap='gray')
# plt.title('Subsampled abs real kspace')
# plt.show()

## TOUCAN
rank = 5
outer = 1
# fun = lambda X: [0, tfrobnorm(X - R) / Rfrob]
fun = lambda X,k: [0, tfrobnorm_array(X - R_sample.array()[:,k,:]) / tfrobnorm_array(
    R_sample.array()[:,k,:])]

Rhat_toucan, U, stats_toucan, tElapsed_toucan = toucan(R_sample,mask,rank,tube=tube,outer=outer,fun=fun,cgtol=1e-9,
                                                     randomOrder=False,verbose=False)

Chat_toucan, U, stats_toucan, tElapsed_toucan = toucan(C_sample,mask,rank,tube=tube,outer=outer,fun=fun,cgtol=1e-9,
                                                     randomOrder=False,verbose=False)

print('Initial R NRMSE: Tensor: {:.8f} '.format(tfrobnorm(R_sample - R) / Rfrob))
print('Initial C NRMSE: Tensor: {:.8f} '.format(tfrobnorm(C_sample - C) / Cfrob))
print('Final R NRMSE: Tensor: {:.8f} '.format(tfrobnorm(Rhat_toucan - R) / Rfrob))
print('Final C NRMSE: Tensor: {:.8f} '.format(tfrobnorm(Chat_toucan - C) / Cfrob))

r_slice_err_toucan = []
c_slice_err_toucan = []

r_slice_err_subsamp = []
c_slice_err_subsamp = []
for i in range(0,n2):
    r_slice_err_toucan.append(tfrobnorm_array(Rhat_toucan.array()[:,i,:] - R.array()[:,i,:]) / tfrobnorm_array(
        R.array()[:,i,]))
    c_slice_err_toucan.append(tfrobnorm_array(Chat_toucan.array()[:, i, :] - C.array()[:, i, :]) / tfrobnorm_array(
        C.array()[:, i, ]))
    r_slice_err_subsamp.append(tfrobnorm_array(R_sample.array()[:, i, :] - R.array()[:, i, :]) / tfrobnorm_array(
        R.array()[:, i, ]))
    c_slice_err_subsamp.append(tfrobnorm_array(C_sample.array()[:, i, :] - C.array()[:, i, :]) / tfrobnorm_array(
        C.array()[:, i, ]))

plt.plot(np.arange(0,len(r_slice_err_toucan)),r_slice_err_toucan,
         np.arange(0,len(c_slice_err_toucan)),c_slice_err_toucan,
         np.arange(0, len(r_slice_err_subsamp)), r_slice_err_subsamp,
         np.arange(0, len(c_slice_err_subsamp)), c_slice_err_subsamp)
plt.legend(['Real TOUCAN','Imag TOUCAN', 'Real Subsamp','Imag Subsamp'])
plt.show()

test = np.log(np.abs(Rhat_toucan.array()[:,vis_idx,:]) + 1e-8)
plt.imshow(test,cmap='gray')
plt.show()

kdata_rec = Rhat_toucan.array() + 1j * Chat_toucan.array()
kdata_rec = np.transpose(kdata_rec,(1,0,2))
im_rec = kspace2im(kdata_rec)

toucan_rec_im = np.abs(im_rec[vis_idx,:,:])
plt.imshow(toucan_rec_im,cmap='gray')
plt.title('TOUCAN reconstructed')
plt.show()

plt.subplot(2,3,1)
plt.imshow(orig_im,cmap='gray')
plt.subplot(2,3,2)
plt.imshow(subsampled_im,cmap='gray')
plt.subplot(2,3,3)
plt.imshow(toucan_rec_im,cmap='gray')
plt.subplot(2,3,4)
plt.imshow(tnn_rec_im,cmap='gray')
plt.subplot(2,3,5)
plt.imshow(tctf_rec_im,cmap='gray')
plt.show()

plt.plot(np.arange(0,len(slice_err_tnn)),slice_err_tnn,np.arange(0,len(slice_err_toucan)),r_slice_err_toucan,
         np.arange(0,len(slice_err_tctf)),slice_err_tctf)
plt.legend(['TNN-ADMM','TOUCAN','TCTF'])
plt.show()