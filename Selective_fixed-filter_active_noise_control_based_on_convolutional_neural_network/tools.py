## This script is used to store some useful tools in acoustic signal processing
import numpy as np
import scipy.signal as signal
import os
import pandas as pd
import scipy.io as sio
import torchaudio

def design_a_weighting(fs):
    """
    * This function is used to design an a-weighting filter according to the sampling rate
    @ params:
    - fs: sampling rate [float]
    @ returns:
    - b: Numerator polynomials of the IIR [ndarray]
    - a: Denominator polynomials of the IIR [ndarray]
    """
    f1 = 20.598997
    f2 = 107.65265
    f3 = 737.86223
    f4 = 12194.217
    A1000 = 1.9997

    numerators = [(2*np.pi * f4)**2 * (10**(A1000/20)), 0, 0, 0, 0]
    denominators = np.convolve([1, 4*np.pi * f4, (2*np.pi * f4)**2],
                               [1, 4*np.pi * f1, (2*np.pi * f1)**2])
    denominators = np.convolve(np.convolve(denominators, 
                                           [1, 2*np.pi * f3]),
                               [1, 2*np.pi * f2])
    
    b, a = signal.bilinear(numerators, denominators, fs)
    return b, a

def a_weighting(waveform, fs):
    b, a = design_a_weighting(fs)
    filtered_wave = signal.lfilter(b,a,waveform)
    return filtered_wave


    
def check_and_count_data_folders(base_path):
    """
    * This function is used to check if the folder exists and count the total number of the files inside this folder
    """
    folders = ['Training_data', 'Testing_data', 'Validating_data']
    folder_data_count = {}
    
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            num_files = len([name for name in os.listdir(folder_path) if 
                             os.path.isfile(os.path.join(folder_path, name))])
            folder_data_count[folder] = num_files
        else:
            folder_data_count[folder] = 'Folder does not exist'

    for folder, count in folder_data_count.items():
        print(f"{folder} contains {count} files." if isinstance(count, int) else f"{folder}:{{count}}")


def loading_paths_from_MAT(folder = r'D:/Coding/Selective_ANC_CNN'
                           ,subfolder = 'Primary and Secondary Path'
                           ,Pri_path_file_name = 'Primary_path.mat'
                           ,Sec_path_file_name ='Secondary_path.mat'):
    """
    * This function is used to load the primary path and secondary path from .mat files
    """
    Primay_path_file, Secondary_path_file = os.path.join(folder, subfolder, Pri_path_file_name), os.path.join(folder,subfolder, Sec_path_file_name)
    Pri_dfs, Secon_dfs = sio.loadmat(Primay_path_file), sio.loadmat(Secondary_path_file)
    Pri_path, Secon_path = Pri_dfs['Pz1'].squeeze(), Secon_dfs['S'].squeeze()
    return Pri_path, Secon_path


def loading_real_wave_noise(sound_name):
    folde_name     = 'D:\Coding\Selective_ANC_CNN\Real_Noise'
    SAMPLE_WAV_SPEECH_PATH = os.path.join(folde_name, sound_name)
    waveform, sample_rate  = torchaudio.load(SAMPLE_WAV_SPEECH_PATH)
    #waveform, sample_rate = _get_sample(SAMPLE_WAV_SPEECH_PATH, 16000)
    resample_rate = 16000
    waveform = resample_wav(waveform, sample_rate, resample_rate)
    return waveform, resample_rate


def resample_wav(waveform, sample_rate,resample_rate):
    resampler = torchaudio.transforms.Resample(sample_rate, resample_rate, dtype=waveform.dtype)
    resampled_waveform = resampler(waveform)
    return resampled_waveform