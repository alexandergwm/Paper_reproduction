import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import scipy.io as sio
import scipy.signal as signal
from scipy.fft import fft, fftfreq, ifft

def minmaxscaler(data):
    min = data.min()
    max = data.max()
    return (data)/(max-min)

class Fixed_filters():
    """
    This class is used to load pre-trained filter from .mat file
    """
    def __init__(self, MATFILE_PATH, fs):
        mat_contents    = sio.loadmat(MATFILE_PATH)
        self.Wc_vectors = mat_contents['Wc_v']
        self.len        = len(self.Wc_vectors)
        self.filterlen  = self.Wc_vectors.shape[1]
        self.Charactors = torch.zeros([self.len, 1, fs], dtype=torch.float)
        self.fs         = fs

        for ii in range(self.len):
            self.Charactors[ii] = self.frequency_charactors_tensor(ii)

    def cancellator(self, classID, Fx, Dir):
        Yt = signal.lfilter(self.Wc_vectors[classID,:], 1, Fx)
        Er = Dir - Yt
        return Er
    
    def frequency_charactors_tensor(self, classID):
        fs = self.fs
        N  = fs + self.filterlen
        xin = np.random.randn(N)
        yout = signal.lfilter(self.Wc_vectors[classID, :], 1, xin)
        yout = yout[self.filterlen:]
        yout = minmaxscaler(yout)
        return torch.from_numpy(yout).type(torch.float).unsqueeze(0)
    

    
    
class Fixed_filter_controller():

    def __init__(self, MAT_FILE, fs):
        self.Wc = self.Load_Pretrained_filters_to_tensor(MAT_FILE)
        Len     = self.Wc.shape[1]
        self.fs = fs 
        self.Xd = torch.zeros(1, Len, dtype= torch.float)
        self.Current_Filter = torch.zeros(1, Len, dtype=torch.float)
    
    def noise_cancellation(self, Dis, Fx, filter_index):
        Erro = torch.zeros(Dis.shape[0])
        j    = 0 
        for ii, dis in enumerate(Dis):
            self.Xd      = torch.roll(self.Xd,1,1)
            self.Xd[0,0] = Fx[ii]
            yt           = self.Current_Filter @ self.Xd.t()
            Erro[ii]     = dis - yt
            if (ii + 1) % self.fs == 0 :
                self.Current_Filter = self.Wc[filter_index[j]]
                j += 1  
        return Erro 
        
    def Load_Pretrained_filters_to_tensor(self, MAT_FILE):
        mat_contents    = sio.loadmat(MAT_FILE)
        Wc_vectors      = mat_contents['Wc_v']
        return  torch.from_numpy(Wc_vectors).type(torch.float)