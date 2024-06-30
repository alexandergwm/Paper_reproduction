import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, ifft
import torch.nn as nn
import torch.optim as optim
from scipy.io import savemat
import pandas as pd

def frequencyband_design(level,fs):
    """
    * This function is used to generate 2^level frequency band
    """
    Num = 2**level
    F_vector = []
    f_start = 20
    f_marge = 20
    width = (fs/2-f_start-f_marge)//Num
    for ii in range(Num):
        f_end = f_start + width
        F_vector.append([f_start, f_end])
        f_start = f_end
    return F_vector, width

class Filter_designer():
    def __init__(self, filter_len, F_vector, fs):
        self.filter_len = filter_len
        self.filter_num = len(F_vector)
        self.wc         = np.zeros((self.filter_num, self.filter_len))
        for i in range(self.filter_num):
            self.wc[i,:] = signal.firwin(self.filter_len, F_vector[i], pass_zero='bandpass', window='hamming', fs=fs)
        
    def __save_mat__(self, FILE_NAME_PATH):
        mdict = {'Wc_v': self.wc}
        savemat(FILE_NAME_PATH, mdict)

def Boardband_Filter_Design_as_Given_Frequencybands(MAT_filename, F_bands, fs):
    """
    * This function is used to design broadband filter by given frequency bands
    """
    Filters = Filter_designer(filter_len=1024, F_vector=F_bands, fs=fs)
    Filters.__save_mat__(MAT_filename)
    print(Filters.filter_num)

def Boardband_Filter_Design_as_Given_F_Levels(MAT_filename, level, fs):
    """
    * This function is used to design boardband filters by given frequency levels
    """
    F_vector = []
    for i in range(level):
        F_vec, _ = frequencyband_design(i, fs)
        F_vector += F_vec

    Filters = Filter_designer(filter_len=1024, F_vector=F_vector, fs=fs)
    Filters.__save_mat__(MAT_filename)
    print(f"There are {Filters.filter_num} pre-train filters has been created")

    #------------------->> Main() <<-----------------------
#-----------------------------------------------------------------------------------
if __name__ == "__main__":
    FILE_NAME_PATH = "Bandlimited_filter.mat"
    fs             = 16000 
    level          = 4 #4 

    F_vector = []
    for i in range(level):
        F_vec, _    = frequencyband_design(i, fs)
        F_vector   += F_vec

    Filters = Filter_designer(filter_len=1024, F_vector= F_vector, fs=fs)
    Filters.__save_mat__(FILE_NAME_PATH)
    print(Filters.filter_num)
    print(F_vector)
