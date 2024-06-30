import numpy as np
import os
import torch
import torchaudio
import scipy.signal as signal
import math
import pandas as pd
from Bcolors import bcolors
import progressbar

def frequencyband_design(level, sample_rate):
    """
    * This function is used to utilized to devide the full frequency band into 
    * several equal frequency components
    @ params: 
    - level
    - sample_rate
    @ returns:
    - F_vector: The vector contains frequency bands
    - width: The width of current frequency band
    """
    Num = 2**level
    F_vector = []
    f_start = 20
    # The redundant frequency components around the maximum frequency
    f_marge = 20         
    # The width of the frequency band
    width = (fs/2 - f_start - f_marge) // Num
    for ii in range(Num):
        f_end = f_start + width
        F_vector.append([f_start, f_end])
        f_start = f_end
    return F_vector, width


class DatasetSheet:
    """
    * This class is used to add the data to the csv file
    """
    def __init__(self, folder, filename):
        self.folder = folder
        self.filename = filename
        try:
            os.makedirs(folder, mode=0o755, exist_ok = True)
        except:
            print('folder exists')
        self.path = os.path.join(folder, filename)

    def add_data_to_file(self, wave_file, class_ID):
        dict         = {'File_path': [wave_file], 'Class_ID': [class_ID]}
        df           = pd.DataFrame(dict)
        
        with open(self.path, mode = 'a') as f:
            df.to_csv(f, header=f.tell()==0)
        
    def flush(self):
        dc       = pd.read_csv(self.path, index_col=0)
        dc.index = range(len(dc))
        dc.to_csv(self.path)


def BandlimitedNoise_generation(f_star, Bandwidth, fs, N):
    # f_star indecats the start of frequency band (Hz)
    # Bandwith denots the bandwith of the boradabnd noise 
    # fs denots the sample frequecy (Hz)
    # N represents the number of point
    len_f = 1024 
    f_end = f_star + Bandwidth
    b2    = signal.firwin(len_f, [f_star, f_end], pass_zero='bandpass', window ='hamming',fs=fs)
    xin   = np.random.randn(N)
    Re    = signal.lfilter(b2,1,xin)
    Noise = Re[len_f-1:]
    #----------------------------------------------------
    return Noise/np.sqrt(np.var(Noise))

def additional_noise(signal, snr_db):
    signal_power     = signal.norm(p=2)
    length           = signal.shape[1]
    additional_noise = np.random.randn(length)
    additional_noise = torch.from_numpy(additional_noise).type(torch.float32).unsqueeze(0)
    noise_power      = additional_noise.norm(p=2)
    snr              = math.exp(snr_db / 10)
    scale            = snr * noise_power / signal_power
    noisy_signal     = (scale * signal + additional_noise) / 2
    return noisy_signal
#-------------------------------------------------------------
# Class : SoundGnereator 
#-------------------------------------------------------------
class SoundGenerator:
    def __init__(self, fs, folder):
        self.fs     = fs 
        self.len    = fs + 1023 
        self.folder = folder 
        self.Num    = 0 
        try: 
            os.mkdir(folder)
        except:
            print("folder exists")
    
    def _construct_(self):
        self.Num  = self.Num + 1 
        f_star    = np.random.uniform(20, 7880, 1)
        bandWidth = np.random.uniform(1,7880-f_star,1)
        f_end     = f_star + bandWidth
        filename  = f'{self.Num}_Frequency_from_'+ f'{f_star[0]:.0f}_to_{f_end[0]:.0f}_Hz.wav'
        filePath  = os.path.join(self.folder, filename)
        noise     = BandlimitedNoise_generation(f_star[0], bandWidth[0], self.fs, self.len)
        noise     = torch.from_numpy(noise).type(torch.float32).unsqueeze(0)
        torchaudio.save(filePath, noise, self.fs)
        return f_star[0], f_end[0], filename
    
    def _construct_A(self):
        self.Num  = self.Num + 1 
        f_star    = np.random.uniform(20, 7880, 1)
        bandWidth = np.random.uniform(1,7880-f_star,1)
        f_end     = f_star + bandWidth
        filename  = f'{self.Num}_Frequency_from_'+ f'{f_star[0]:.0f}_to_{f_end[0]:.0f}_Hz_A.wav'
        filePath  = os.path.join(self.folder, filename)
        noise     = BandlimitedNoise_generation(f_star[0], bandWidth[0], self.fs, self.len)
        noise     = torch.from_numpy(noise).type(torch.float32).unsqueeze(0)
        snr_db    = np.random.uniform(3, 60, 1)
        noise     = additional_noise(noise, snr_db)
        torchaudio.save(filePath, noise, self.fs)
        return f_star[0], f_end[0], filename
    
    def _balance_construct(self, Fre_noise_vector):
        self.Num  = self.Num + 1 
        filename  = f'{self.Num}_Frequency_from_'+ f'{Fre_noise_vector[0]:.0f}_to_{Fre_noise_vector[1]:.0f}_Hz.wav'
        filePath  = os.path.join(self.folder, filename)
        noise     = BandlimitedNoise_generation(Fre_noise_vector[0], Fre_noise_vector[1]-Fre_noise_vector[0], self.fs, self.len)
        noise     = torch.from_numpy(noise).type(torch.float32).unsqueeze(0)
        torchaudio.save(filePath, noise, self.fs)
        return filename
    
    def _balance_construct_A(self, Fre_noise_vector):
        self.Num  = self.Num + 1 
        filename  = f'{self.Num}_Frequency_from_'+ f'{Fre_noise_vector[0]:.0f}_to_{Fre_noise_vector[1]:.0f}_Hz.wav'
        filePath  = os.path.join(self.folder, filename)
        noise     = BandlimitedNoise_generation(Fre_noise_vector[0], Fre_noise_vector[1]-Fre_noise_vector[0], self.fs, self.len)
        noise     = torch.from_numpy(noise).type(torch.float32).unsqueeze(0)
        snr_db    = np.random.uniform(3, 60, 1)
        noise     = additional_noise(noise, snr_db)
        torchaudio.save(filePath, noise, self.fs)
        return filename

def similarty_ratio(f1_min, f1_max, f2_min, f2_max):
    """
    * This function is used to calculate the overlap ratio between two frequency bands
    * and get the similarity between them
    """
    if (f1_min <= f2_min):
        if (f1_max <= f2_min):
            return 0
        elif (f2_min <= f1_max) & (f1_max <= f2_max):
            return (f1_max-f2_min)/(f2_max-f1_min)
        else:
            return (f2_max-f2_min)/(f1_max-f1_min)
    else:
        if (f2_max <= f1_min):
            return 0
        elif (f1_min <= f2_max)&(f2_max <= f1_max):
            return (f2_max-f1_min)/(f1_max-f2_min)
        else:
            return (f1_max-f1_min)/(f2_max-f2_min)


class ClassID_Calculator:
    """
    * This function will call the SimilarityRatio to find the most similar pre-defined frequency band with 
    * input frequency band, and get the corresponding class index
    """ 
    def __init__(self, levels):
        self.f_vector = levels
        self.len = len(self.f_vector)

    def _get_ID_(self, f_low, f_high):
        SimilartyRatio = []
        for ii in range(self.len):
            SimilartyRatio.append(similarty_ratio(f_low, f_high, self.f_vector[ii][0], self.f_vector[ii][1]))
        ID = SimilartyRatio.index(max(SimilartyRatio))
        return ID, SimilartyRatio
    


def generating_dataset_as_given_frequencybands(N_sample_each_class, F_bands, Folder_name):
    """
    * This function is used to generate dataset as given frequency band
    """
    file_name = "Index.csv"
    Datasheet = DatasetSheet(Folder_name, file_name)
    Generator = SoundGenerator(fs=16000, folder = Folder_name)

    Frequency_noise_band, Frequency_ID = generating_balance_sampleset_frequency_band_vector(Frequency_band=F_bands, Sample_set_number=N_sample_each_class)
    Frequency_noise_band_A, Frequency_ID_A = generating_balance_sampleset_frequency_band_vector(Frequency_band=F_bands, Sample_set_number=N_sample_each_class)
    print(bcolors.RED + f'Each sample set has {len(Frequency_ID) + len(Frequency_ID_A)} !!!' + bcolors.ENDC)

    bar = progressbar.ProgressBar(maxval=len(Frequency_ID), \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    ii = 0
    bar.start()
    for Frequency_noise_vector, Frequency_ID_element, Frequency_noise_vector_A, Frequency_ID_element_A in zip(Frequency_noise_band,Frequency_ID,Frequency_noise_band_A,Frequency_ID_A):
        filepath = Generator._balance_construct(Fre_noise_vector=Frequency_noise_vector)
        Datasheet.add_data_to_file(filepath, Frequency_ID_element)

        filepath = Generator._balance_construct_A(Fre_noise_vector=Frequency_noise_vector_A)
        Datasheet.add_data_to_file(filepath, Frequency_ID_element_A)
        ii += 1
        bar.update(ii)
    Datasheet.flush()
    bar.finish()


def generating_balance_sampleset_frequency_band_vector(Frequency_band, Sample_set_number):
    """
    * This function is used to generate the balance frequency band vector
    * it mainly focus on generating the same number samples for each class to satisfy the balance of dataset
    """
    Max_number = Sample_set_number
    ID_calculator = ClassID_Calculator(Frequency_band)
    Class_count = np.zeros(len(Frequency_band))
    Class_num = len(Frequency_band)
    Frequency_noise_band = []
    Frequency_ID = []

    continue_flag = True
    while continue_flag:
        F_band = np.sort(np.random.uniform(20, 7880, 2))
        if F_band[0] == F_band[1]:
            continue
        ID,_ = ID_calculator._get_ID_(f_low=F_band[0], f_high=F_band[1])
        if Class_count[ID] < Max_number:
            Frequency_noise_band.append(F_band)
            Frequency_ID.append(ID)
            Class_count[ID] += 1

        if np.sum(Class_count == Max_number) == Class_num:
            continue_flag = False
        else:
            continue_flag = True
    print(bcolors.OKGREEN + f' Have created {Class_num} balance frequency band for datast !!!' + bcolors.ENDC)
    return Frequency_noise_band, Frequency_ID




if __name__ == '__main__':
    level = 4
    fs = 16000
    F_band = []
    F_vector, width = frequencyband_design(level,fs)
    # print(F_vector)
    for i in range(level):
        temp_vector, _ = frequencyband_design(i, fs)
        F_band += temp_vector

    Folder_name_list_of_dataset = ['Training_data', 'Validating_data', 'Testing_data']
    print(40000 // len(F_band))     # 2666
    N_sample_list = [(40000//len(F_band)), (1000//len(F_band)), (1000//len(F_band))]

    for folder_name, N_sample_element in zip(Folder_name_list_of_dataset, N_sample_list):
        generating_dataset_as_given_frequencybands(N_sample_each_class=N_sample_element, F_bands=F_band, Folder_name=folder_name)
        print(bcolors.OKCYAN + f'Has finihsed {folder_name} !!!!' + bcolors.ENDC)