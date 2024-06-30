import torch
import os
import numpy as np
from torch import nn
import scipy.signal as signal
from ONED_CNN_pre import ONED_CNN_Predictor
from canceling_noise_fixed_filter import Fixed_filters
from design_filter import Boardband_Filter_Design_as_Given_Frequencybands, frequencyband_design

def Casting_single_time_length_of_training_noise(filter_training_noise, fs):
    """
    This function is used to truncate the input signal into 1s (16000 samples)
    """
    assert filter_training_noise.dim() == 3, "The dimension of the training noise should be 3 !!!"
    print(filter_training_noise[:,:,:fs].shape)
    return filter_training_noise[:,:,:fs]

def Casting_multiple_time_length_of_primary_noise(primary_noise,fs):
    assert  primary_noise.shape[0] == 1, 'The dimension of the primary noise should be [1 x samples] !!!'
    cast_len = primary_noise.shape[1] - primary_noise.shape[1]%fs
    return primary_noise[:,:cast_len]


class Control_filter_Index_predictor(ONED_CNN_Predictor):
    """
    * This function is used to use ONED_CNN net to predict the control filter index according to the primary input noise
    """
    def __init__(self, MODEL_PTH, device, filter_training_noise, fs):

        ONED_CNN_Predictor.__init__(self, MODEL_PTH, device)
        # Checking the length of the training noise
        # Here the truncated 1s samples are used to predict the adaptive filter index
        assert filter_training_noise.dim() == 3, "The dimensions of the training noise should be 3"
        assert filter_training_noise.shape[2] % fs == 0, "The length of the training noise sample should be 1 second"
        # Detach the information of the training noise
        self.frequency_charactors_tensor = filter_training_noise
        self.len_of_class                = filter_training_noise.shape[0]
        self.fs                          = fs

    def predic_ID(self, noise):
        similarity_ratio = []
        for ii in range(self.len_of_class):
            similarity_ratio.append(self.cosSimilarity_minmax(noise, self.frequency_charactors_tensor[ii]))
        index = np.argmax(similarity_ratio)
        return index
    
    def predic_ID_vector(self, primary_noise):
        # Checking the length of the primary noise
        assert primary_noise.shape[0] == 1, 'The dimension of the primary noise should be [1 x samples] !!!'
        assert primary_noise.shape[1] % self.fs == 0, 'The length of the primary noise is not an integral multiple of fs.'
        # Computing how many seconds the primary noise contained
        Time_len  = int(primary_noise.shape[1]/self.fs)
        print(f'The primary noise has {Time_len} seconds')
        # Building the matric of the primary noise [times x 1 x fs]
        primary_noise_vectors = primary_noise.reshape(Time_len, self.fs).unsqueeze(1)

        # Implementing the noise classification for each frame with length of 1 second
        ID_vector =[]
        for ii in range(Time_len):
            ID_vector.append(self.predic_ID(primary_noise_vectors[ii]))
        
        return ID_vector


def Control_filter_selection(MODEL_PTH_type=1, fs=16000, num_frequency_bands=4, Primary_noise = None):
    if num_frequency_bands == 4:
        Frequency_band = [[20, 550], [450, 1200], [1000, 2700], [2500, 4500], [4400, 7980]]
        # Creating the pre-trained band filter for 5 different frequency bands
        Filter_mat_name = "Boardband_filter_from_5frequencybands.mat"

        if not os.path.exists(Filter_mat_name):
            Boardband_Filter_Design_as_Given_Frequencybands(MAT_filename=Filter_mat_name, F_bands=Frequency_band, fs=16000)
        else:
            print("Data of " + Filter_mat_name + 'is existed !!!')
    elif num_frequency_bands == 15:
        F_vector = []
        Filter_mat_name = "Boardband_filter_from_15frequencybands.mat"
        for i in range(4):
            F_vec, _ = frequencyband_design(i, fs)
            F_vector += F_vec
        if not os.path.exists(Filter_mat_name):
            Boardband_Filter_Design_as_Given_Frequencybands(MAT_filename=Filter_mat_name, F_bands=F_vector, fs=16000)
        else:
            print("Data of" + Filter_mat_name + "is existed !!!")

    Fixed_control_filter = Fixed_filters(MATFILE_PATH=Filter_mat_name, fs=fs)
    Charactors = Casting_single_time_length_of_training_noise(Fixed_control_filter.Charactors, fs=fs)
    # cnn model path
    if MODEL_PTH_type == 1:
        MODEL_PTH = "D:/Coding/Gavin/Selective_ANC_CNN/feedforwardnet_1D.pth"



    # the decision module will be running in the cpu
    device = "cpu"

    Pre_trained_control_filter_ID_predictor = Control_filter_Index_predictor(MODEL_PTH=MODEL_PTH,
                                                                             device = device,
                                                                             filter_training_noise = Charactors,
                                                                             fs=fs)
    
    Primary_noise = Casting_multiple_time_length_of_primary_noise(Primary_noise, fs=fs)

    Id_vector = Pre_trained_control_filter_ID_predictor.predic_ID_vector(Primary_noise)

    return Id_vector
