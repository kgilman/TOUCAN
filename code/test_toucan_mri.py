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
#     c = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(kdata, axes=1), axis=1), axes=1)
#     im_rec = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(c, axes=2), axis=2), axes=2)
    return np.abs(im_rec)

### Brain MRI data

real_filepath = '/Users/kgilman/Desktop/t-SVD/real_mri.mat'
imag_filepath = '/Users/kgilman/Desktop/t-SVD/imag_mri.mat'

real_data = sio.loadmat(real_filepath)['real_mri']
imag_data = sio.loadmat(imag_filepath)['imag_mri']

kdata = real_data + 1j*(imag_data)
#
kdata = np.transpose(kdata,(2,0,1))
orig_im = kspace2im(kdata)

plt.imshow(orig_im[10,:,:],cmap='gray')
plt.show()

vis_idx = 69
## Read in the images


# plt.figure(figsize=(20,20))
plt.subplot(1,2,1)
plt.imshow(orig_im[vis_idx],cmap='gray')
plt.title('Original magnitude image')

plt.subplot(1,2,2)
test = np.log(np.abs(kdata[vis_idx,:,:]) + 1e-13)
plt.imshow(test,cmap='gray')
plt.title('Magnitude kspace')
plt.rcParams.update({'font.size': 22})
name = 'original_and_kspace.eps'
plt.savefig(name)
plt.show()

### Form the real and imag tensors
kdata_orig = kdata.copy()
kdata = np.transpose(kdata,(1,0,2))

R = np.real(kdata)
Rmean = np.mean(R)

C = np.imag(kdata)
Cmean = np.mean(C)

# R -= Rmean
# C -= Cmean

R = Tensor(R)
C = Tensor(C)

n1,n2,n3 = R.shape()

### Generate k-space sampling mask

tube = False
rho = 0.5  # percentage of missing entries
# rho = 0.4  # percentage of missing entries

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
# subsampled_im = np.abs(im_rec[vis_idx,:,:])
subsampled_im = np.abs(im_rec)

# plt.figure(figsize=(15,12))
plt.subplot(1,2,1)
plt.imshow(subsampled_im[vis_idx,:,:],cmap='gray')
plt.title('Reconstruction from subsampled')
plt.subplot(1,2,2)
plt.imshow(np.log(np.abs((kdata_rec[vis_idx,:,:])) + 1e-8),cmap='gray')
plt.title('Magnitude of subsampled kspace')
plt.rcParams.update({'font.size': 22})
name = 'subsampled_and_kspace_tube_' + str(tube) + '_' + str(int(rho*100)) + '.eps'
plt.savefig(name)
plt.show()

## TOUCAN
# rank = 5
rank = 1
outer = 1
# fun = lambda X: [0, tfrobnorm(X - R) / Rfrob]
fun = lambda X,k: [0, tfrobnorm_array(X.array()[:,k,:] - R.array()[:,k,:]) / tfrobnorm_array(R.array()[:,k,:])]

Rhat_toucan, U, stats_toucan, tElapsed_toucan = toucan(R_sample,mask,rank,tube=tube,mode='online',outer=outer,
                                                       fun=fun,cgtol=1e-1,randomOrder=False,verbose=True)

# fun = lambda X,k: [0, tfrobnorm_array(X.array()[:,k,:] - C.array()[:,k,:]) / tfrobnorm_array(C.array()[:,k,:])]
# Chat_toucan, U, stats_toucan, tElapsed_toucan = toucan(C_sample,mask,rank,tube=tube,mode='online',outer=outer,
#                                                        fun=fun,cgtol=1e-9,randomOrder=False,verbose=True)
#
# print('Initial R NRMSE: Tensor: {:.8f} '.format(tfrobnorm(R_sample - R) / Rfrob))
# print('Initial C NRMSE: Tensor: {:.8f} '.format(tfrobnorm(C_sample - C) / Cfrob))
print('Final R NRMSE: Tensor: {:.8f} '.format(tfrobnorm(Rhat_toucan - R) / Rfrob))
# print('Final C NRMSE: Tensor: {:.8f} '.format(tfrobnorm(Chat_toucan - C) / Cfrob))

print('TOUCAN time: {:.4f}'.format(tElapsed_toucan))