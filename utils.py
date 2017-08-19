import os
import pickle
import numpy as np
import librosa

def create_dir(directory):
    if not os.path.exists(directory):
        print("Creating directory : {}".format(directory))
        os.makedirs(directory)
    else:
        print("Already existing directory : {}".format(directory))

def create_file(path):
    if not os.path.isfile(path):
        with open(path,'w') as f:
            pass

def append_to_file(path,data):
    """
        Add data on the last line of path
    """
    with open(path,'a') as f:
        f.write(data+'\n')

def delete_file_contents(path):
    """
        Delete file and write file agian
    """
    with open(path,'w'):
        pass

def pickle_load(path):
    f = open(path, 'rb')
    temp = pickle.load(f)
    f.close()
    return temp

def pickle_store(content, path):
    f = open(path, 'wb')
    pickle.dump(content, f)
    f.close()

def complex_dot(v1, v2):
    return np.dot(v1, np.conj(v2))

def clip_by_value(x, v_max = 1, v_min = 0):
    if x>v_max:
        return v_max
    if x<v_min :
        return v_min
    return x

def onset_times(sound, sampling_rate):
    '''
    Arg :
        sound - 1D array song
        sampling_rate - sampling_rate
    Return : 
        1D array moments(times) of onsets (seconds)
    '''
    return librosa.frames_to_time(librosa.onset.onset_detect(y=sound, sr=sampling_rate), sr=sampling_rate)

def normalize(x):
    '''
    Arg : 
        x - numpy 1D array
    Return :
        1D array normalized to be 1
    '''
    sum_ = np.sum(x)
    return x/sum_

def DTFS(sound, islog = False, compressed_ratio = 100):
    '''
    Arg :
        sound : 1D array
        islog : boolean
        compressed_ratio : int

    Return :
        perform DTFS(Discrete time fourier series)
        if  islog == True : normalize(log(1 + compressed_ratio*DTFS))
        else : normalize(DTFS)
    '''
    period = len(sound)
    fourier = np.fft.fft(sound) # DTFT of sound

    # Get half of DTFT
    abs_fourier_half = np.zeros(period, dtype = np.float32) 
    for k in range(int(period/2)):
        abs_fourier_half[k] = abs(fourier[k])
    
    if islog:
        return normalize(np.log(1 + compressed_ratio*abs_fourier_half))
    else :
        return normalize(abs_fourier_half)
