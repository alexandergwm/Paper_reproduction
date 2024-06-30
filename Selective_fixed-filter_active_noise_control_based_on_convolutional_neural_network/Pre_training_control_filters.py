from tools import loading_paths_from_MAT
from generating_Disturbance import DistDisturbance_reference_generation_from_Fvector
import numpy as np
from FxLMS_algorithm import FxLMS_algorithm, train_fxlms_algorithm
import matplotlib.pyplot as plt
from design_filter import frequencyband_design
from scipy.io import savemat



def save_mat__(FILE_NAME_PATH, Wc):
    mdict = {'Wc_v': Wc}
    savemat(FILE_NAME_PATH, mdict)

def main():
    fs = 16000
    T = 30
    Len_control = 1024

    num_control_filters = 15
    if (num_control_filters == 5):
        Frequency_band = [[20, 550], [450, 1200], [1000, 2700],[2500, 4500],[4400, 7980]]
        FILE_NAME_PATH = "Control_filter_from_5frequencies.mat"
    if (num_control_filters == 15):
        level = 4
        Frequency_band = []
        for i in range(level):
            F_vec, _ = frequencyband_design(i, fs)
            Frequency_band += F_vec
        FILE_NAME_PATH = "Control_filter_from_15frequencies.mat"
        
    # loading the primary and secondary path
    Pri_path, Secon_path = loading_paths_from_MAT()
    # training the control filters from the defined frequency band
    num_filters = len(Frequency_band)
    Wc_matrix = np.zeros((num_filters, Len_control), dtype=float)

    for ii, F_vector in enumerate(Frequency_band):
        Dis, Fx = DistDisturbance_reference_generation_from_Fvector(fs=fs, T=T, f_vector=F_vector, Pri_path=Pri_path, Sec_path=Secon_path)
        controller = FxLMS_algorithm(Len_control)

        Error = train_fxlms_algorithm(Model=controller, Ref=Fx, Disturbance=Dis)
        Wc_matrix[ii] = controller._get_coeff_()

    save_mat__(FILE_NAME_PATH, Wc_matrix)

if __name__ == "__main__":
    main()