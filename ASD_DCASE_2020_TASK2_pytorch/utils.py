import os
import pickle
import time
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import librosa
import sys



def setup(params):
    """
    Sets up the environment for training by creating directories for model checkpoints
    and logging, saving configuration parameters, and initializing a tensorboard summary writer.
    Args:
        params (dict): Dictionary containing the configuration parameters.
    Returns:
        tuple: A tuple containing the path to the checkpoints folder, output folder and the tensorboard summary writer instance.
    """

    print('You are using the following configuration: \n\n')
    for key, value in params.items():
        print(key, ': ', value)

    # create dir to save model checkpoints
    reference = f"{params['net_type']}_{time.strftime('_%Y%m%d_%H%M%S')}"
    checkpoints_dir = os.path.join(params['checkpoints_dir'], reference)
    os.makedirs(checkpoints_dir, exist_ok=True)

    # save the all the config/hyperparams to a pickle file
    pickle_filepath = os.path.join(str(checkpoints_dir), 'config.pkl')
    pickle_file = open(pickle_filepath, 'wb')
    pickle.dump(params, pickle_file)

    # create a tensorboard summary writer for logging and visualization
    log_dir = os.path.join(params['log_dir'], reference)
    os.makedirs(log_dir, exist_ok=True)
    summary_writer = SummaryWriter(log_dir=str(log_dir))

    # create output folder to save the predictions
    result_dir = os.path.join(params['result_dir'], reference)
    os.makedirs(result_dir, exist_ok=True)

    return checkpoints_dir, result_dir, summary_writer


def get_log_mel_spectrogram(filename,
                            n_mels=64,
                            n_fft=1024,
                            hop_length=512,
                            power=2.0):
    wav, sampling_rate = file_load(filename)
    mel_spectrogram = librosa.feature.melspectrogram(y=wav,
                                                     sr=sampling_rate,
                                                     n_fft=n_fft,
                                                     hop_length=hop_length,
                                                     n_mels=n_mels,
                                                     power=power)
    log_mel_spectrogram = 20.0 / power * np.log10(mel_spectrogram + sys.float_info.epsilon)
    return log_mel_spectrogram

def file_load(wav_name, mono=False):
    """
    load .wav file.

    wav_name : str
        target .wav file
    sampling_rate : int
        audio file sampling_rate
    mono : boolean
        When load a multi channels file and this param True, the returned data will be merged for mono data

    return : np.array( float )
    """
    try:
        return librosa.load(wav_name, sr=None, mono=mono)
    except:
        logger.error("file_broken or not exists!! : {}".format(wav_name))
