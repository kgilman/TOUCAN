import numpy as np
import matplotlib.pyplot as plt

path = '/Users/kgilman/Desktop/WTD_upload/Toluene_200/'
mypath = '/Users/kgilman/Desktop/WTD_upload/Toluene_200/L4/'
strfile = '/Users/kgilman/Desktop/WTD_upload/Toluene_200/L4/201106060617_board_setPoint_500V_fan_setPoint_060_mfc_setPoint_Toluene_200ppm_p7'

from os import listdir
from os.path import isfile, join
myfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

def find_nearest(array, values):
    indices = np.abs(np.subtract.outer(array, values)).argmin(0)
    return indices

datalist = []
with open(strfile,'r') as file:
    for line in file:
        datalist.append(line.split('\t'))
    file.close()


i = 0

# values = np.arange(0,260000,100)    #10 Hz
# data = np.zeros((len(myfiles),2600,71))
# outfile = path + 'L4array_10hz'

values = np.arange(0,260000,200)    #5 Hz
data = np.zeros((len(myfiles),1300,71))
outfile = path + 'L4array_5hz'

# values = np.arange(0,260000,1000)    #1 Hz
# data = np.zeros((len(myfiles),260,71))
# outfile = path + 'L4array_1hz'

data_means = np.zeros((len(myfiles),71))
# data_norms = np.zeros((len(myfiles),71))
data_norms = np.zeros((len(myfiles)))
for file in myfiles:
    datalist = []
    with open(mypath + file,'r') as f:
        for line in f:
            datalist.append(line.split('\t'))
    f.close()

    #Sample at desired frequency
    data_array = np.array(datalist)

    time_idx = data_array[:,0].astype(int)
    sample_idx = find_nearest(time_idx,values)
    data_array = data_array[sample_idx,:]

    #Filter the columns
    data_array = data_array[:,-1-80:-2]
    data_array = data_array.astype(int)

    data_idx = np.where(data_array[1,:]!=1)[0]
    data_array = data_array[:,data_idx]

    #Convert to float, subtract time-series mean, normalize by Frob norm
    data_array = data_array.astype(float)
    means = np.mean(data_array,axis=0)
    data_array -= means
    norms = np.linalg.norm(data_array,'fro')
    # norms = np.linalg.norm(data_array, axis=0)
    # norms = np.max(data_array,axis=0)

    data_array /= norms

    data[i,:,:] = data_array
    data_means[i,:] = means
    # data_norms[i,:] = norms
    data_norms[i] = norms
    i += 1

np.savez(outfile,data,data_means,data_norms)

print("I'm done")