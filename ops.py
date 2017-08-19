import tensorflow as tf
import numpy as np
import pandas as pd
import os
import csv
import glob
import librosa
from pydub import AudioSegment
import requests
import urllib
from bs4 import BeautifulSoup
from tqdm import tqdm

from utils import create_dir
from utils import pickle_load
from utils import pickle_store
from utils import clip_by_value
from utils import DTFS
from utils import onset_times
from utils import complex_dot

from configs import SC_URL, SR, MUSIC_DIR, PCA_DIR, FEATURES_DIR, ONSET_DIR, ONSET_MODEL_NAME

def song_path(dir_name, index):
    '''
    Args :
        dir_name - string
            path of directory
        index - int
            used for index of song
    return :
        song_path_ - string
            path of the song
        ex) dir_name = hi, index = 1  => hi/001.wav
    '''
    song_name = str(index).zfill(3)+'.wav'
    song_path_ = os.path.join(dir_name, song_name)
    return song_path_

def mp3_to_wav(mp3filename, wavfilename):
    '''
    Args:
        mp3filename - string
        wavfilename - tring
    Process:
        change mp3 from mp3filename to wav as wavfilename
    '''
    sound = AudioSegment.from_mp3(mp3filename)
    print("Converting from {} to {}".format(mp3filename, wavfilename))
    sound.export(wavfilename, format="wav")
    os.remove(mp3filename)

def download_link(link):
    '''
    Download the link

    Args :
        link - string
            soundcloud link(here) can be link from others
    '''
    print("Downloading {}".format(link))
    os.system("youtube-dl " + link)

def add_music_link(links, query):
    '''
    Arg:
        links - list
            for append new links
        query - string
            Used for searching in soundcloud
    '''
    response = requests.get(SCURL+query)
    soup = BeautifulSoup(response.text,'html.parser')
    for link in soup.select('a'):
        href = link['href']
        if href[0]=='/' and len(href)!=1:
            if not href[1:7]=='search' and not href[1:8]=='popular':
                href='https://soundcloud.com'+link['href']
                if href not in links:
                    links.append(href)

def query2wav(file_path):
    '''
    Arg :
        file_path - string
            filename for queries  ex) lstm.txt
            filename should be txt
    Process :
        Download the song when queries are used to search music at soundcloud
    '''
    # links : every links for music when queried with query on soundcloud
    links = []
    with open(file_path, 'r') as f:
        # line : query(string)
        for line in f:
            add_music_link(links, line)
    links.sort()
    # wave_dir generation
    # just remove .txt
    wav_dir = file_path[:-4]
    create_dir(wav_dir)
    # link download
    count = 0
    for link in links:
        download_link(link)
        mp3_lists = glob.glob('*.mp3')
        for mp3_name in mp3_lists:
            try :
                count+=1
                wav_name = song_path(wav_dir, count)
                mp3_to_wav(mp3_name, wav_name)
            except FileNotFoundError:
                print("FileNotFoundError : {}".format(mp3_name))

def music_load(music_path, start=0.0, end=60.0):
    '''
    Arg
        music_path - string
            path of music to load
        start - float
            start point of music to extract(sec)
        end - float
            end point of music to extract(sec)

    return
        y - 1D array
            mono sound data from music path fron start(sec) to end(sec)
        sr - int
            sampling_rate defaults to be SR(=22050)
    '''
    print("Music is loaded on {}".format(music_path))
    y, sr = librosa.load(path = music_path, sr=SR, offset = start, duration = end - start)
    return y, sr

def get_pca_basis_path(clip_sec):
    '''
    Arg
        clip_sec - float
            clip_sec for making spectogram during pca
    '''
    return os.path.join(PCA_DIR, "pca_basis_{}.pkl".format(clip_sec))

def pca_generator(clip_sec = 0.1, MUSIC_DIR = MUSIC_DIR):
    '''
    Need to be updated to make pca according to the spectrum randomly sampled from music
    Currently spectogram sampled from 0-60 sec of song

    Args
        clip_sec - float
            clip length of sound(sec)
        MUSIC_DIR - string
            > default to be MUSIC_DIR
            > path of wav files directory to load wav file
    Process
        Store following in pickle_path
            content
                w : eigen value
                v : eigen vector
    '''
    create_dir(PCA_DIR)
    pickle_path = get_pca_basis_path(clip_sec)
    window = int(SR*clip_sec)
    nfft = window #FFT window size
    win_len = window # win_len <= nfft
    hop_len = int(window/4)
    duration = 60 #sec

    print("nfft : {}, win_len : {}, hop_len : {}".format(nfft, win_len, hop_len))

    spec = []
    wav_files = glob.glob(os.path.join(MUSIC_DIR, '*.wav'))
    for wav_file in tqdm(wav_files):
        y, sr = librosa.load(path = wav_file,
                             sr = SR,
                             duration=duration)
        D = librosa.core.stft(y, n_fft = nfft, win_length = win_len, hop_length= hop_len)
        nfreqs, ntimes = D.shape

        for i in range(ntimes):
            spec.append(D[:,i])

    spec = np.array(spec)
    spec_cov = np.cov(spec.T)

    print("Spectogram vector : {}".format(spec.shape))
    print("Covariance of spectogram : {}".format(spec_cov.shape))

    w, v = np.linalg.eig(spec_cov) # w : eigen value v : eigen vector

    content = {'w' : w, 'v' : v}
    pickle_store(content, pickle_path)
    print("Pickle file is stored in {}".format(pickle_path))

def get_pca_features(y, npca_comp=40, tss=0.1):
    '''
    Get pca features

    Args
        y - 1D array
            mono sound_data
        npca_comp - int
            number of pca components needed to extract pca features
        tss - float
            clip length of sound(sec)
    return :
        pca_features - 2D array(ntseg, npca_comp)
    '''
    # length of sound(y)
    sound_len = len(y)
    # length of time segment
    tseg_len = int(tss*SR)
    # number of time segments in the music
    ntseg = sound_len//tseg_len

    pca_basis_path = get_pca_basis_path(tss)

    try :
        content = pickle_load(pca_basis_path)
    # if pca basis with tss doesn't exist, generate pca basis with tss
    except FileNotFoundError:
        print("FileNotFoundError : {}".format(pca_basis_path))
        pca_generator(clip_sec = tss)
        content = pickle_load(pca_basis_path)

    w = content['w'] # eigen value
    v = content['v'] # eigen vector

    print("Number of pca vectors : {}".format(len(w)))

    window = int(SR*tss)
    nfft = window
    win_len = window # win_len<=nfft
    hop_len = window

    D = librosa.core.stft(y,
                          n_fft=nfft,
                          win_length=win_len,
                          hop_length=hop_len)

    D_T = D.T
    ntime, nfreq = D_T.shape

    pca_features = np.zeros((ntseg, npca_comp))
    for i in range(min(ntime, ntseg)):
        for j in range(npca_comp):
            pca_features[i][j] = abs(complex_dot(D_T[i], v[j]))

    return pca_features

def get_onset_classify_index(y, model_name = ONSET_MODEL_NAME, forward = 0.03, backward = 0.07, islog = True, comp = 1.0):
    '''
    feature extraction with model

    Args
        y - 1D array
            mono sound_data
        model_name - string
            name of model
        forward - float
            clip from onset - forward
        backward - float
            clip to onset + backward
        islog - bool
            log on the spectrum on DTFS
        comp - float
            compression ratio
    return :
        exclude_label - exclude_label
            loaded from info.pkl
        sound_type - dict
            loaded from info.pkl
        features - 2D array(len(y), len(sound_type))
    '''
    MODEL_PATH = os.path.join(ONSET_DIR, 'models_save/{}/f{}b{}log{}comp{}/'.format(model_name,
                                                              forward,
                                                              backward,
                                                              islog,
                                                              comp))

    info = pickle_load(os.path.join(MODEL_PATH, 'info.pkl'))
    exclude_label = info['exclude_label']
    sound_type = info['sound_type']

    nclasses = len(sound_type)
    sound_len = len(y)

    onsets = onset_times(y, SR)
    n_onsets = len(onsets)

    sound_classify = np.zeros((len(onsets), nclasses))
    features = np.zeros((sound_len, nclasses))

    # sound_classify update
    with tf.Session() as sess:
        # Neural network restoration
        restorer = tf.train.import_meta_graph(os.path.join(MODEL_PATH, 'model.meta'))
        restorer.restore(sess, os.path.join(MODEL_PATH, 'model'))
        freqs = tf.get_collection("input")[0]
        index = tf.get_collection("output")[0]

        if model_name == 'DNN_dropconnect':
            istrain = tf.get_collection("istrain")[0]

        # Apply neural network
        feed_dict = {}
        if model_name == 'DNN_dropconnect':
            feed_dict[istrain] = False

        for i in range(n_onsets):
            standard = int(onsets[i]*SR)
            clip_from = standard - int(forward*SR)
            clip_to = standard + int(backward*SR)
            if clip_from>=0 and clip_to<sound_len:
                dtfs = DTFS(sound = y[clip_from:clip_to],
                            islog = islog,
                            compressed_ratio = comp)
                feed_dict[freqs] =  np.reshape(dtfs ,[1,-1])
                classify_index = np.reshape(sess.run(index, feed_dict = feed_dict), [-1])
                for j in range(nclasses):
                    sound_classify[i][j] = classify_index[j]

    tf.reset_default_graph()

    # features update
    for i in range(len(onsets)):
        standard = int(onsets[i]*SR)
        clip_from = standard - int(forward*SR)
        clip_to = standard + int(backward*SR)
        if clip_from>=0 and clip_to<sound_len:
            for j in range(clip_from, clip_to):
                for k in range(nclasses):
                    features[j][k] += sound_classify[i][k]

    # clip the value of features from 0 to 1
    for i in range(sound_len):
        for j in range(nclasses):
            features[i][j] = clip_by_value(features[i][j], 1, 0)

    return exclude_label, sound_type, features

def get_reduced_onset_features(y, tss = 0.1, model_name = ONSET_MODEL_NAME, forward = 0.03, backward = 0.07, islog = True, comp = 1.0):
    '''
    Invariant over tss
    Remove the features over excluded label, and change to get the features per tss

    Args
        y - 1D array
            mono sound_data
        tss - float
            time segment size
        model_name - string
            name of model
        forward - float
            clip from onset - forward
        backward - float
            clip to onset + backward
        islog - bool
            log on the spectrum on DTFS
        comp - float
            compression ratio
    return
        selected_sound_type - list

        reduced_onset_features - 2D array((ntseg, selected_nclasses))
    '''
    exclude_label, sound_type, onset_features = get_onset_classify_index(y,
                                                                   model_name = model_name,
                                                                   forward = forward,
                                                                   backward = backward,
                                                                   islog = islog,
                                                                   comp = comp)
    sound_len, nclasses = onset_features.shape

    selected_onset_features = []
    selected_sound_type = []

    for i in range(nclasses):
        if i in exclude_label:
            continue
        selected_onset_features.append(onset_features[:,i])
    selected_onset_features = np.array(selected_onset_features)
    selected_onset_features_t = selected_onset_features.T

    for i in sound_type.keys():
        if i in exclude_label:
            continue
        selected_sound_type.append(sound_type[i])

    # length of time segment
    tseg_len = int(tss*SR)
    # number of time segments in the music
    ntseg = sound_len//tseg_len
    # number of selected type
    selected_nclasses = len(selected_sound_type)

    reduced_onset_features = np.zeros((ntseg, selected_nclasses))
    for i in range(ntseg):
        for j in range(selected_nclasses):
            reduced_onset_features[i][j] = np.average(selected_onset_features_t[i*tseg_len:(i+1)*tseg_len, j])

    return selected_sound_type, reduced_onset_features

def get_features(y, tss = 0.1, npca_comp = 30, model_name=ONSET_MODEL_NAME):
    '''
    Arg
        y - 1D array
            mono sound data
        tss - float
            time segment size(sec) default to be 0.1
        npca_comp - int
            number of pca components be used to extract features
    return
        header - 1D array(string)
            header information for each columns in features
        features - 2D array(float)
            shape (ntseg, npca_comp + len(selected_sound_type))
    '''
    # length of sound(y)
    sound_len = len(y)
    # length of time segment
    tseg_len = int(tss*SR)
    # number of time segments in the music
    ntseg = sound_len//tseg_len

    print("Total {} datas with sampling rates(={})".format(sound_len, SR))
    print("Each time segment is {} sec with {} datas".format(tss, tseg_len))
    print("Total {} segments".format(ntseg))

    pca_features = get_pca_features(y, npca_comp=npca_comp, tss=tss)
    print("PCA features : {}".format(pca_features.shape))

    selected_sound_type, reduced_onset_features = get_reduced_onset_features(y, tss = tss, model_name = model_name, forward = 0.03, backward = 0.07, islog = True, comp = 1.0)
    print("Onset features : {}".format(reduced_onset_features.shape))

    header = ["pca{}".format(i+1) for i in range(npca_comp)] + [stype for stype in selected_sound_type]
    features = np.concatenate((pca_features, reduced_onset_features), axis = 1)
    print("Total {} features are extracted".format(features.shape[1]))
    return header, features

def music2csv(music_path, csv_path = os.path.join(FEATURES_DIR, 'features.csv'), start = 0, end = 1, tss = 0.1, npca_comp = 30):
    '''
    Args
        music_path - string
            path of music to extract features
        csv_path - string
            path of csv to store index
        start - float
            start point of music to extract(sec)
        end - float
            end point of music to extract(sec)
        npca_comp - int
            number of pca components be used to extract features
    '''
    create_dir(FEATURES_DIR)
    y, sr = music_load(music_path = music_path,  start = start, end = end)
    header_info, features = get_features(y, tss=tss, npca_comp = npca_comp)
    ntseg = features.shape[0]

    df = pd.DataFrame(features)
    df.index = np.linspace(tss, ntseg*tss, ntseg)
    df.to_csv(csv_path, header=header_info)

    print("Features from {} are stored in {}\n".format(music_path, csv_path))
