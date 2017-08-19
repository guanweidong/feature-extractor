import os

#=================== PATH =================#

# Project directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Directory to save music files (wav)
MUSIC_DIR = os.path.join(BASE_DIR, 'music_dir')

# Txt format to store queries
QUERY_PATH = os.path.join(BASE_DIR, 'query.txt')

# Directory to save features (csv)
FEATURES_DIR = os.path.join(BASE_DIR, 'features')

## @maestrojeong: Add details

# PCA
PCA_DIR = os.path.join(BASE_DIR, 'pca_dataset')

# ONSET
ONSET_DIR = os.path.join(BASE_DIR, 'onset')

ONSET_MODEL_NAME = 'DNN'

#=================== Music =================#

# Sampling_rate
SR = 22050

#=================== URLs =================#

# Default url for soundcloud
SC_URL = 'https://soundcloud.com/search/sounds?q='
